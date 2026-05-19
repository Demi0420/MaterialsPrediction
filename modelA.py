import json
import os
import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pymatgen.core import Structure, Lattice as PmgLattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from sklearn.model_selection import train_test_split
import csv
import tqdm
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import copy
from datetime import datetime
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pyxtal.symmetry import Group
from pyxtal.lattice import Lattice as PyxLattice
from pyxtal import pyxtal
from collections import Counter


# ======================
# CGCNN Feature Extraction Components
# ======================

def gaussian_expansion(distance, centers, gamma=40.0):
    """
    Apply Gaussian basis expansion to a distance.
    Args:
        distance (float or np.ndarray): Distance value(s).
        centers (np.ndarray): Array of center values for Gaussians.
        gamma (float): Width parameter for Gaussians.
    Returns:
        np.ndarray: Gaussian expanded distance of shape [len(centers)].
    """
    return np.exp(-gamma * (distance - centers) ** 2)

def build_cgcnn_graph(structure: Structure, z2index, cutoff=6.0, max_num_nbr=12, 
                      radius_step=6.0/127, max_radius=6.0):
    """
    Construct a crystal graph in CGCNN style for a pymatgen Structure.
    Returns:
      node_fea: np.ndarray of shape [N_atoms] with atomic numbers.
      edge_index: np.ndarray of shape [2, E] with pairs of neighbor indices.
      edge_fea: np.ndarray of shape [E, M] with Gaussian distance features.
    """
    # 1. Node features: atomic numbers for each site
    # atomic_nums = np.array([site.specie.Z for site in structure], dtype=np.int64)  # [N]
    atomic_nums = np.array([z2index[site.specie.Z] for site in structure], dtype=np.int64)  # [N]

    # 2. Find neighbors within cutoff
    center_indices, neighbor_indices, _, distances = structure.get_neighbor_list(r=cutoff)
    N = len(structure)
    # adjacency lists
    adjacency = [[] for _ in range(N)]
    dist_list = [[] for _ in range(N)]
    for c_idx, n_idx, dist in zip(center_indices, neighbor_indices, distances):
        adjacency[c_idx].append(n_idx)
        dist_list[c_idx].append(dist)
    # Prepare Gaussian distance centers
    num_centers = int(max_radius / radius_step) + 1
    gauss_centers = np.linspace(0, max_radius, num_centers)
    # 3. Build edges with features
    edge_src = []
    edge_dst = []
    edge_features = []
    for i in range(N):
        # sort neighbors by distance and keep up to max_num_nbr
        nbrs = adjacency[i]
        nbr_dists = dist_list[i]
        sorted_nbrs = sorted(zip(nbrs, nbr_dists), key=lambda x: x[1])[:max_num_nbr]
        for j, dist in sorted_nbrs:
            edge_src.append(i)
            edge_dst.append(j)
            edge_fea = gaussian_expansion(dist, gauss_centers)
            edge_features.append(edge_fea)
    if len(edge_features) == 0:
        # Handle isolated atom (no neighbors within cutoff)
        edge_features = np.zeros((0, num_centers), dtype=float)
    else:
        edge_features = np.stack(edge_features, axis=0).astype(np.float32)  # [E, M]
    edge_index = np.stack([edge_src, edge_dst], axis=0).astype(np.int64)   # [2, E]
    return atomic_nums, edge_index, edge_features

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor):
    """Sum `src` values into `out` at positions given by index along dimension `dim`."""
    out.index_add_(dim, index, src)
    return out

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor):
    """Compute mean by summing `src` at index positions and dividing by counts."""
    count = torch.zeros_like(out)
    count.index_add_(dim, index, torch.ones_like(src))
    out.index_add_(dim, index, src)
    out = out / (count + 1e-8)
    return out

class AtomEmbedding(nn.Module):
    """Embedding layer for atomic numbers."""
    def __init__(self, max_atom_num=85, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_atom_num, embedding_dim=embed_dim)
    def forward(self, x: torch.LongTensor):
        # x shape: [N_atoms]
        return self.embedding(x)

class CGCNNConv(nn.Module):
    """One interaction layer of CGCNN (simplified)."""
    def __init__(self, atom_fea_dim: int, edge_fea_dim: int):
        super().__init__()
        # Linear layer to transform concatenated (atom_i, atom_j, edge) to messages
        self.fc_full = nn.Linear(2 * atom_fea_dim + edge_fea_dim, 2 * atom_fea_dim)
    def forward(self, atom_fea: torch.Tensor, edge_index: torch.Tensor, edge_fea: torch.Tensor):
        # atom_fea: [N, atom_fea_dim], edge_index: [2, E], edge_fea: [E, edge_fea_dim]
        src, dst = edge_index  # source and destination atom indices for each edge
        # Gather features for each edge
        atom_src = atom_fea[src]  # [E, atom_fea_dim]
        atom_dst = atom_fea[dst]  # [E, atom_fea_dim]
        # Concatenate source, destination atom features and edge features
        edge_in = torch.cat([atom_src, atom_dst, edge_fea], dim=1)  # [E, 2*atom_fea_dim + edge_fea_dim]
        # Transform and split into gate and update (core) signals
        edge_out = self.fc_full(edge_in)  # [E, 2 * atom_fea_dim]
        gate, core = torch.chunk(edge_out, chunks=2, dim=1)  # each [E, atom_fea_dim]
        gate = torch.sigmoid(gate)
        core = torch.tanh(core)
        message = gate * core  # gated message
        # Aggregate messages for each destination atom
        agg = torch.zeros_like(atom_fea)
        agg = scatter_add(message, dst, dim=0, out=agg)
        # Update atom features
        new_atom_fea = F.softplus(atom_fea + agg)
        return new_atom_fea

class CGCNNFeatureExtractor(nn.Module):
    """
    CGCNN network to produce a crystal latent vector (graph embedding + optional extra features).
    Used here for extracting latent representations of structures.
    """
    def __init__(self, 
                 atom_fea_dim=64, 
                 edge_fea_dim=128, depth=3, 
                 num_atom_types=85, 
                 use_extra_fea=True, extra_fea_dim=11):
        super().__init__()
        self.use_extra_fea = use_extra_fea
        # Embedding for atomic numbers
        self.embed = AtomEmbedding(max_atom_num=num_atom_types, embed_dim=atom_fea_dim)
        # Stacking multiple graph convolution layers
        self.convs = nn.ModuleList(CGCNNConv(atom_fea_dim, edge_fea_dim) for _ in range(depth))
        # Note: final latent dimension = atom_fea_dim + extra_fea_dim (if use_extra_fea) or atom_fea_dim.
    def forward(self, node_fea: torch.Tensor, edge_index: torch.Tensor, edge_fea: torch.Tensor, 
                crystal_atom_idx: torch.Tensor, extra_fea: torch.Tensor = None):
        """
        Forward pass to get crystal embedding.
        Args:
            node_fea: [N_atoms] Long tensor of atomic numbers for all atoms in batch.
            edge_index: [2, E] Long tensor of edge connections (global indexing for batch).
            edge_fea: [E, edge_fea_dim] Float tensor of edge features.
            crystal_atom_idx: [N_atoms] Long tensor indicating which crystal each atom belongs to.
            extra_fea: [Batch, extra_fea_dim] Float tensor of extra features for each crystal.
        Returns:
            crystal_latent: [Batch, atom_fea_dim + extra_fea_dim] latent vector for each crystal.
        """
        # Atom embedding
        # print(node_fea.shape)
        atom_fea = self.embed(node_fea)  # shape [total_atoms, atom_fea_dim]
        # Graph convolutions
        for conv in self.convs:
            atom_fea = conv(atom_fea, edge_index, edge_fea)
        # Graph-level pooling (mean over atoms for each crystal in the batch)
        num_graphs = crystal_atom_idx.max().item() + 1  # number of graphs in batch
        graph_emb = scatter_mean(atom_fea, crystal_atom_idx, dim=0, 
                                 out=torch.zeros(num_graphs, atom_fea.shape[1], device=atom_fea.device))
        # If extra features are provided, concatenate them to graph embedding
        if self.use_extra_fea and extra_fea is not None:
            graph_emb = torch.cat([graph_emb, extra_fea], dim=1)
        return graph_emb

class StructureDataset(Dataset):
    """
    Dataset for structures, providing latent vector and condition vector for each sample.
    """
    def __init__(
            self, 
            df: pd.DataFrame, 
            cond_cols: list, 
            extra_feature_cols: list, 
            max_atoms: int,
            z2index: dict,
            a_min=3.0, a_max=18.0, alpha_min=45.0, alpha_max=135.0,
            cond_mean=None,
            cond_std=None
        ):
        """
        Args:
            df: DataFrame with structure objects and properties.
            cond_cols: List of column names for conditioning properties (e.g., melting point, density, formation energy).
            extra_feature_cols: List of column names for extra features to include in latent.
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.cond_cols = cond_cols
        self.extra_feature_cols = extra_feature_cols
        self.max_atoms = max_atoms
        self.z2index = z2index

        self.a_min = a_min
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Compute mean and std for extra features (for normalization)
        self.extra_mean = self.df[extra_feature_cols].mean()
        self.extra_std = self.df[extra_feature_cols].std().replace({0: 1.0})

        # Compute mean and std for condition features (for normalization)
        # Compute mean and std for condition features (for normalization)
        if cond_mean is None:
            self.cond_mean = self.df[cond_cols].mean()
        else:
            self.cond_mean = cond_mean

        if cond_std is None:
            self.cond_std = self.df[cond_cols].std().replace({0: 1.0})
        else:
            self.cond_std = cond_std

        # Initialize CGCNN feature extractor for latent computation
        # self.cgcnn = CGCNNFeatureExtractor(use_extra_fea=True, extra_fea_dim=len(extra_feature_cols))

        # (Optional) If a pre-trained CGCNN model is available, you could load weights here.
        self.device = torch.device('cpu')  # default compute device for CGCNN

    
    def normalize_lattice_params(self, a, b, c, alpha, beta, gamma):
        """Clamp and map to [0,1]."""
        a_ = max(min(a, self.a_max), self.a_min)
        b_ = max(min(b, self.a_max), self.a_min)
        c_ = max(min(c, self.a_max), self.a_min)
        alpha_ = max(min(alpha, self.alpha_max), self.alpha_min)
        beta_  = max(min(beta,  self.alpha_max), self.alpha_min)
        gamma_ = max(min(gamma, self.alpha_max), self.alpha_min)

        a_norm = (a_ - self.a_min) / (self.a_max - self.a_min)
        b_norm = (b_ - self.a_min) / (self.a_max - self.a_min)
        c_norm = (c_ - self.a_min) / (self.a_max - self.a_min)

        alpha_norm = (alpha_ - self.alpha_min) / (self.alpha_max - self.alpha_min)
        beta_norm  = (beta_  - self.alpha_min) / (self.alpha_max - self.alpha_min)
        gamma_norm = (gamma_ - self.alpha_min)/ (self.alpha_max - self.alpha_min)
        return np.array([a_norm, b_norm, c_norm, alpha_norm, beta_norm, gamma_norm], dtype=np.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Retrieve structure and properties
        row = self.df.iloc[idx]
        structure: Structure = row["structure"]
        
        try:
            analyzer = SpacegroupAnalyzer(structure)
            sg_number = analyzer.get_space_group_number()
            symm_dataset = analyzer.get_symmetry_dataset()
            wyckoff_symbols = symm_dataset.wyckoffs         # ['a', 'b', 'c', ...]
            wyckoff_sites = symm_dataset.equivalent_atoms   # [0, 0, 1, 2, ...]

            wyckoff_species = [site.specie.symbol for site in structure.sites]

            # Count the occurrences of each Wyckoff letter to obtain the multiplicity
            counter = Counter(wyckoff_symbols)  # e.g., {'a': 4, 'b': 8}
            wyckoff_labels = [f"{counter[sym]}{sym}" for sym in wyckoff_symbols]

            # Use coordinates as free parameters
            max_param_dim = 6
            wyckoff_params = [site.frac_coords.tolist() + [0.0]*(max_param_dim - 3) for site in structure.sites]

            # Pad or truncate to the length of max_atoms
            wp_labels_pad = ['0'] * self.max_atoms
            wp_params_pad = [ [] for _ in range(self.max_atoms) ]
            wp_species_pad = ['X'] * self.max_atoms  # ‘X’ indicates a vacancy

            for j in range(min(len(wyckoff_labels), self.max_atoms)):
                wp_labels_pad[j] = wyckoff_labels[j]
                wp_params_pad[j] = wyckoff_params[j]
                wp_species_pad[j] = wyckoff_species[j]

            sg_id = torch.tensor(sg_number, dtype=torch.long)
            wp_labels = wp_labels_pad
            wp_params = wp_params_pad
            wp_species = wp_species_pad

        except Exception as e:
            # Set to the default value if the analysis fails
            sg_id = torch.tensor(1, dtype=torch.long)
            wp_labels = ['0'] * self.max_atoms
            wp_params = [ [] for _ in range(self.max_atoms) ]
            wp_species = ['X'] * self.max_atoms


        # Build graph representation for the structure
        node_fea_np, edge_index_np, edge_fea_np = build_cgcnn_graph(structure, self.z2index)
        # node_fea, edge_index, edge_fea = build_cgcnn_graph(structure, self.z2index)
        # print(f"node_fea min: {node_fea_np.min()}, max: {node_fea_np.max()}")
        mn, mx = node_fea_np.min(), node_fea_np.max()
        if mx >= 85:
            print(f"[DEBUG] Sample {idx}: node_fea max={mx}, min={mn}, => out of range!")
            
        # Convert to torch tensors on the specified device
        node_fea = torch.tensor(node_fea_np, dtype=torch.long, device=self.device)
        # print("node_fea.shape =", node_fea.shape, "node_fea=", node_fea)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
        edge_fea = torch.tensor(edge_fea_np, dtype=torch.float32, device=self.device)
        
        # Create crystal index tensor (all atoms in this structure have index 0, since one structure per sample)
        N = node_fea.shape[0]
        crystal_atom_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        
        # 2) extra_fea (normalized)
        extra_vals = []
        for col in self.extra_feature_cols:
            val = (row[col] - self.extra_mean[col]) / (self.extra_std[col] + 1e-8)
            extra_vals.append(val)
        extra_vals = torch.tensor(extra_vals, dtype=torch.float32).unsqueeze(0)  # shape [1, extra_dim]

        # 3) cond (normalized)
        cond_vals = []
        for c in self.cond_cols:
            val = (row[c] - self.cond_mean[c]) / (self.cond_std[c] + 1e-8)
            cond_vals.append(val)
        cond_vals = torch.tensor(cond_vals, dtype=torch.float32)

        # 4) prepare GT lattice params
        latt = structure.lattice
        a, b, c = latt.a, latt.b, latt.c
        alpha, beta, gamma = latt.alpha, latt.beta, latt.gamma
        lattice_norm = self.normalize_lattice_params(a, b, c, alpha, beta, gamma)  # shape [6]

        # 5) fractional coords + types
        frac_coords = structure.frac_coords
        atom_types = [site.specie.Z for site in structure]
        coords_pad = np.zeros((self.max_atoms, 3), dtype=np.float32)
        types_pad  = np.zeros((self.max_atoms,),   dtype=np.int64)

        N_i = len(atom_types)
        frac_coords_clamped = np.clip(frac_coords, 0, 1)
        for j in range(min(N_i, self.max_atoms)):
            coords_pad[j] = frac_coords_clamped[j]
            Z_j = atom_types[j]
            if Z_j in self.z2index:
                types_pad[j] = self.z2index[Z_j]  # map Z->(1..N), 0 is empty
            else:
                types_pad[j] = 0

        coords_pad_ts = torch.tensor(coords_pad, dtype=torch.float32)
        types_pad_ts  = torch.tensor(types_pad,  dtype=torch.long)

        return {
            "node_fea": node_fea,
            "edge_index": edge_index,
            "edge_fea": edge_fea,
            "crystal_atom_idx": crystal_atom_idx,
            "extra_fea": extra_vals,  # shape [1, extra_dim]
            "cond": cond_vals,        # shape [cond_dim]
            "lattice_label": torch.tensor(lattice_norm, dtype=torch.float32),  # shape [6]
            "coords_label": coords_pad_ts,   # [max_atoms,3]
            "types_label": types_pad_ts,     # [max_atoms]
            "sg_id": sg_id,                        # Space group number（int）
            "wyckoff_labels": wp_labels,           # String list of length max_atoms
            "wyckoff_params": wp_params,           # Free parameter list of length max_atoms
            "wyckoff_species": wp_species          # Element symbol list of length max_atoms
        }


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        atom_fea_dim=64,
        edge_fea_dim=128,
        depth=3,
        max_atom_num=20,
        num_atom_types=85,
        extra_fea_dim=11,
        latent_dim=64+11,  # 64 graph embed + 11 extra = 75
        decoder: nn.Module = None,
        prop_pred: nn.Module = None
    ):
        super().__init__()
        self.encoder = CGCNNFeatureExtractor(
            atom_fea_dim=atom_fea_dim,
            edge_fea_dim=edge_fea_dim,
            depth=depth,
            num_atom_types=num_atom_types,
            use_extra_fea=True,
            extra_fea_dim=extra_fea_dim
        )
        self.decoder = decoder  # StructureDecoder
        self.prop_pred = prop_pred  # PropertyPredictor

    def forward(
        self,
        node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
    ):
        """
        Returns:
          latent: [batch=1, latent_dim]
          lat_dec_out = decoder(latent)
          prop_out = prop_pred(latent)  (if not None)
        """
        # Assume batch_size=1 per structure if you don't do a bigger multi-structure batch
        latent = self.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)  # shape [1, latent_dim]
        self.decoder_latent = latent
        # decode
        lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params = self.decoder(latent)
        prop_out = None
        if self.prop_pred is not None:
            prop_out = self.prop_pred(latent)
        return lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params, prop_out
    


def collate_fn(batch):
    node_feas = []
    edge_indices = []
    edge_feas = []
    crystal_atom_idxs = []
    extra_feas = []
    conds = []  # will be data["cond"]
    lattice_labels = []
    coords_labels = []
    types_labels = []

    sg_ids = []              # [B]
    wyckoff_labels_batch = []    # [B, max_atoms]
    wyckoff_params_batch = []    # [B, max_atoms, variable_length]
    wyckoff_species_batch = []   # [B, max_atoms]



    atom_offset = 0
    for i, data in enumerate(batch):
        n_atoms = data['node_fea'].size(0)

        node_feas.append(data['node_fea'])  # [N_i]
        edge_indices.append(data['edge_index'] + atom_offset)
        edge_feas.append(data['edge_fea'])
        crystal_atom_idxs.append(torch.full((n_atoms,), i, dtype=torch.long))
        extra_feas.append(data['extra_fea'].squeeze(0))  # ensure shape [extra_dim]

        conds.append(data['cond'])  # Use condition vector for property target
        lattice_labels.append(data['lattice_label'])
        coords_labels.append(data['coords_label'])
        types_labels.append(data['types_label'])

        sg_ids.append(data['sg_id'])  # [1]
        wyckoff_labels_batch.append(data['wyckoff_labels'])       # List[str]
        wyckoff_params_batch.append(data['wyckoff_params'])       # List[List[float]]
        wyckoff_species_batch.append(data['wyckoff_species'])     # List[str]


        atom_offset += n_atoms

    return {
        "node_fea": torch.cat(node_feas, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_fea": torch.cat(edge_feas, dim=0),
        "crystal_atom_idx": torch.cat(crystal_atom_idxs, dim=0),
        "extra_fea": torch.stack(extra_feas),   # [B, extra_dim]
        "cond": torch.stack(conds),
        "lattice_label": torch.stack(lattice_labels),
        "coords_label": torch.stack(coords_labels),
        "types_label": torch.stack(types_labels),
        "sg_id": torch.stack(sg_ids),                               # [B]
        "wyckoff_labels": wyckoff_labels_batch,                     # List[List[str]], length = B
        "wyckoff_params": wyckoff_params_batch,                     # List[List[List[float]]]
        "wyckoff_species": wyckoff_species_batch                    # List[List[str]]
    }

def compute_class_weights(dataset, num_atom_types):
    # Count the type labels of all samples
    all_types = []
    for sample in dataset:
        all_types.append(sample["types_label"].numpy())  # shape: [max_atoms]
    all_types = np.concatenate(all_types)                # shape: [N_total_atoms]
    
    # Count the occurrences of each type
    type_counts = np.bincount(all_types, minlength=num_atom_types)

    # Compute weights ∝ 1 / count
    class_weights = 1.0 / (type_counts + 1e-6)

    # Normalize
    class_weights = class_weights / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)


def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean', ignore_index=None):
    """
    logits: [N, num_classes]
    targets: [N] (long)
    alpha: [num_classes] or scalar; if not None, apply class weighting
    gamma: focusing parameter
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [N, C]
    probs = log_probs.exp()                    # [N, C]

    # Create one-hot encoding for targets
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

    # Gather log_probs and probs at target classes
    log_pt = (log_probs * targets_one_hot).sum(dim=-1)  # shape [N]
    pt = log_pt.exp()

    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            at = alpha.gather(0, targets)
        else:
            at = torch.tensor(alpha).to(logits.device)
            at = at.gather(0, targets)
        log_pt = log_pt * at

    focal_term = (1 - pt) ** gamma
    loss = - focal_term * log_pt

    if ignore_index is not None:
        mask = targets != ignore_index
        loss = loss[mask]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    

def atom_pairwise_dist_penalty(coords, mask, threshold=0.5):
    B, N, _ = coords.shape
    loss = 0.0
    for b in range(B):
        valid = mask[b].squeeze(-1) > 0
        c = coords[b][valid]  # [N_valid, 3]
        if c.size(0) <= 1: continue
        dist = torch.cdist(c, c, p=2)  # [N_valid, N_valid]
        triu_mask = torch.triu(torch.ones_like(dist), diagonal=1)
        pairwise_dists = dist[triu_mask == 1]
        too_close = pairwise_dists[pairwise_dists < threshold]
        if too_close.numel() > 0:
            loss += ((threshold - too_close)**2).mean()
    return loss / B


def nonzero_type_penalty(types_logits, target_min=4):
    pred_probs = F.softmax(types_logits, dim=-1)  # [B, N, T]
    max_probs = pred_probs.max(dim=-1).values  # [B, N]
    nonzero_mask = (max_probs > 0.1).float()
    num_nonzero = nonzero_mask.sum(dim=1)  # [B]
    return F.relu(target_min - num_nonzero).mean()



def compute_wyckoff_losses(
    sg_logits, pred_wp_logits, pred_wp_params, 
    batch_data, label2index, device, max_param_dim=6
):
    """
    Compute three supervised losses related to Wyckoff symmetry:
    1. Space group classification loss (sg_loss)
    2. Wyckoff label classification loss (wyckoff_label_loss)
    3. Wyckoff free parameter regression loss (wyckoff_param_loss)

    Args:
        batch_data: A batch returned by the DataLoader, containing sg_id, wyckoff_labels, wyckoff_params
        model: The current model; the decoder is expected to output sg_fc, pred_wp_logits, and pred_wp_params
        label2index: A dictionary mapping Wyckoff labels (e.g., '4a') to integers
        device: torch.device
        max_param_dim: Maximum number of degrees of freedom (default is 6)

    Returns:
        sg_loss: Space group classification loss
        wyckoff_label_loss: Wyckoff label classification loss
        wyckoff_param_loss: Wyckoff parameter regression loss
    """
    # === Ground Truth ===
    sg_ids = batch_data["sg_id"].to(device)                # [B]
    wyckoff_labels = batch_data["wyckoff_labels"]          # List[List[str]]
    wyckoff_params = batch_data["wyckoff_params"]          # List[List[List[float]]]

    B = len(sg_ids)

    # 1. Space group classification loss
    sg_loss = F.cross_entropy(sg_logits, sg_ids - 1)  # sg_id ranges from 1 to 230, so subtract 1 for indexing

    # 2. Wyckoff label classification loss
    wp_label_indices = []
    for label_list in wyckoff_labels:
        index_list = [label2index.get(lbl, 0) for lbl in label_list]  # Map to 0 if the label is not found
        wp_label_indices.append(torch.tensor(index_list, dtype=torch.long))
    wyckoff_label_tensor = torch.stack(wp_label_indices).to(device)  # [B, max_atoms]

    B, N = wyckoff_label_tensor.shape
    wp_logits_2d = pred_wp_logits.view(B * N, -1)
    wp_targets_1d = wyckoff_label_tensor.view(-1)
    wyckoff_label_loss = F.cross_entropy(wp_logits_2d, wp_targets_1d)

    # 3. Wyckoff parameter regression loss (MSE)
    wp_param_tensors = []
    for param_list in wyckoff_params:
        padded = []
        for p in param_list:
            p = p[:max_param_dim] + [0.0] * (max_param_dim - len(p))
            padded.append(p)
        wp_param_tensors.append(torch.tensor(padded, dtype=torch.float32))
    wp_param_target = torch.stack(wp_param_tensors).to(device)  # [B, max_atoms, max_param_dim]

    wyckoff_param_loss = F.mse_loss(pred_wp_params, wp_param_target)
    
    return sg_loss, wyckoff_label_loss, wyckoff_param_loss

def train_encoder_decoder(
    model: EncoderDecoderModel,
    dataset: StructureDataset,
    label2index,
    epochs=20,
    batch_size=1,  # if you handle one structure per step for simplicity
    learning_rate=1e-4,
    device=torch.device('cpu'),
    lattice_w=10.0, coord_w=10.0, type_w=2.0, prop_w=2.0,
    patience=20,
    num_atom_types=85,
    save_path="best_encoder_decoder.pth"
):
    """
    Example: end-to-end training for CGCNN encoder + StructureDecoder + optional prop_pred.
    We do a simple single-sample batch for clarity. 
    If you want real batch, you'll need a custom collate_fn to combine multiple structures into one big graph.
    """

    class_weights = compute_class_weights(dataset, num_atom_types)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.to(device)
    model.train()

    # Collect parameters from encoder, decoder, maybe prop_pred:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    lattice_std_history = []
    epoch_type_stats = []

    for epoch in range(1, epochs+1):
        # epoch_losses = []
        epoch_total_loss = 0.0
        epoch_lattice_loss = 0.0
        epoch_coord_loss = 0.0
        epoch_type_loss = 0.0
        # epoch_prop_loss = 0.0
        epoch_sg_loss = 0.0
        epoch_wyckoff_label_loss = 0.0
        epoch_wyckoff_param_loss = 0.0

        num_batches = 0

        for batch_data in dataloader:
            # batch_data is a dict with the keys we returned from __getitem__
            node_fea = batch_data["node_fea"].to(device)               # [N_atoms]
            # print("node_fea.shape =", node_fea.shape, "node_fea=", node_fea)
            edge_index = batch_data["edge_index"].to(device)           # [2, E]
            edge_fea = batch_data["edge_fea"].to(device)               # [E, edge_fea_dim]
            crystal_atom_idx = batch_data["crystal_atom_idx"].to(device) # [N_atoms]
            extra_fea = batch_data["extra_fea"].to(device)             # [1, extra_dim]
            cond = batch_data["cond"].to(device)                       # [cond_dim]
            lattice_label = batch_data["lattice_label"].to(device)     # [6]
            coords_label = batch_data["coords_label"].to(device)       # [max_atoms,3]
            types_label = batch_data["types_label"].to(device)

            # forward
            lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params, prop_out = model(
                node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
            )

            # compute losses
            # 1) lattice_loss: compare lattice_norm to lattice_label
            #    (both in [0,1], see dataset code)
            # lattice_loss = F.mse_loss(lattice_norm.squeeze(0), lattice_label)
            if torch.isnan(lattice_norm).any():
                print("[Warning] Lattice norm has NaN, skipping.")
                continue

            lattice_loss = F.mse_loss(lattice_norm, lattice_label)
            lattice_std = torch.std(lattice_norm, dim=0).mean()
            # print(f"[Batch] Lattice diversity: std = {lattice_std.item():.4f}")

            lattice_std_history.append(lattice_std.item())

            def pairwise_var_loss(x):
                # encourage embeddings to be spread out in batch
                diff = x.unsqueeze(1) - x.unsqueeze(0)
                dist = (diff ** 2).sum(-1)
                return -dist.mean()
            diversity_penalty = - 0.01 * pairwise_var_loss(lattice_norm)

            # 2) coords_loss
            mask = (types_label > 0).unsqueeze(-1).float()  # shape [max_atoms,1]
            # coord_diff = (coords.squeeze(0) - coords_label) * mask
            coord_diff = (coords - coords_label) * mask
            coord_loss = (coord_diff**2).sum() / (mask.sum()*3.0 + 1e-8)
            dist_loss = atom_pairwise_dist_penalty(coords, mask)


            # === 2.5) coord diversity loss: encourage atom spread ===
            def coord_pairwise_diversity(coords, mask):
                """
                coords: [B, N, 3]
                mask: [B, N, 1]
                """
                B, N, _ = coords.shape
                loss = 0.0
                for b in range(B):
                    valid = mask[b].squeeze(-1) > 0  # shape [N]
                    c = coords[b][valid]             # [N_valid, 3]
                    if c.size(0) <= 1: continue
                    diff = c.unsqueeze(0) - c.unsqueeze(1)  # [N, N, 3]
                    dist = (diff**2).sum(-1) + 1e-8         # Add a stability term to avoid sqrt(0)
                    dist = torch.sqrt(dist)                # [N, N]
                    triu_mask = torch.triu(torch.ones_like(dist), diagonal=1)
                    pairwise_dists = dist[triu_mask == 1]
                    loss += -pairwise_dists.mean()
                return loss / B

            coord_diversity_loss = coord_pairwise_diversity(coords, mask)
            # print(f"[Batch] Coord diversity loss = {coord_diversity_loss.item():.4f}")

            # 3) type_loss (cross_entropy)
            # types_logits: [batch=1, max_atoms, num_atom_types]
            #   => squeeze(0) => [max_atoms, num_atom_types]
            # types_label: [max_atoms]
            type_loss = 0.0
            type_entropy_bonus = 0.0
            if types_logits is not None:
                # pred_types_2d = types_logits.squeeze(0)  # [max_atoms, num_atom_types]
                pred_types_2d = types_logits.view(-1, types_logits.size(-1))
                
                valid_mask = types_label.view(-1) > -1

                # === (1) Force class_weights[0] to be very small to prevent the model from learning type=0 ===
                class_weights[0] = 0.05  # Padding type is unimportant, assign it a very small weight

                type_loss = focal_loss(
                    logits=pred_types_2d[valid_mask],
                    targets=types_label.view(-1)[valid_mask],
                    alpha=class_weights.to(device), 
                    gamma=1.5                   # Adjustable — the larger it is, the more focused the model becomes
                )
                
                # === (3) Suppress type=0 in logits to prevent predicting 0 during inference ===
                type_count_loss = nonzero_type_penalty(types_logits, target_min=4)
                true_atom_counts = (types_label > 0).sum(dim=1).float()  # [B]
                pred_types = types_logits.argmax(dim=-1)
                pred_atom_counts = (pred_types != 0).sum(dim=1).float()
                count_loss = F.mse_loss(pred_atom_counts, true_atom_counts)

                
                # 3.5) type prediction distribution analysis (debugging collapse)
                with torch.no_grad():
                    pred_types_2d = torch.clamp(pred_types_2d, -30, 30)  # Clamp the logits to prevent numerical explosion
                    pred_type_probs = F.softmax(pred_types_2d, dim=-1)
                    entropy = - (pred_type_probs * torch.log(pred_type_probs + 1e-8)).sum(dim=-1)
                    # pred_type_probs = F.softmax(pred_types_2d, dim=-1)  # [B*N, num_atom_types]
                    # entropy = - (pred_type_probs * torch.log(pred_type_probs + 1e-8)).sum(dim=-1)  # [N]
                    mean_entropy = entropy.mean()
                
                    pred_types = pred_type_probs.argmax(dim=-1).cpu().numpy()  # [B*N]
                    # true_types = types_label.view(-1).cpu().numpy()

                    # pred_type_counts = np.bincount(pred_types, minlength=types_logits.size(-1))
                    # true_type_counts = np.bincount(true_types, minlength=types_logits.size(-1))
                
                type_entropy_bonus = mean_entropy
                # print(f"[Batch] Type entropy: {mean_entropy.item():.4f}")
                # print(f"[Batch] Predicted type distribution: {pred_type_counts}")
                # print(f"[Batch] True      type distribution: {true_type_counts}")



            # 4) property loss
            # prop_loss = 0.0
            # if prop_out is not None:
                # prop_out shape [1, cond_dim], cond shape [cond_dim]
                # => compare
                # prop_loss = F.mse_loss(prop_out.squeeze(0), cond)
            #      prop_loss = F.mse_loss(prop_out, cond)


            # === Wyckoff Loss (Space group, position, parameter) ===
            try:
                # z is stored in model.decoder_latent during the model’s forward pass
                z = model.decoder_latent
                sg_logits = model.decoder.sg_fc(z)

                sg_loss, wyckoff_label_loss, wyckoff_param_loss = compute_wyckoff_losses(
                    sg_logits, pred_wp_logits, pred_wp_params,
                    batch_data, label2index, device
                )
            except Exception as e:
                print("[Warning] Wyckoff loss computation failed:", e)
                sg_loss = torch.tensor(0.0, device=device)
                wyckoff_label_loss = torch.tensor(0.0, device=device)
                wyckoff_param_loss = torch.tensor(0.0, device=device)

            total_loss = (
                lattice_w * lattice_loss + 
                diversity_penalty + 
                coord_w * coord_loss + 
                0.2 * coord_diversity_loss +
                2.0 * dist_loss + 
                type_w * type_loss +
                -0.1 * type_entropy_bonus +
                0.5 * type_count_loss +
                1.0 * count_loss + 
                1.0 * sg_loss +
                1.0 * wyckoff_label_loss + 
                0.5 * wyckoff_param_loss
                # prop_w * prop_loss
            )
            
            if torch.isnan(total_loss):
                print("[Warning] total_loss is NaN. Skipping this batch.")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # epoch_losses.append(total_loss.item())
            # loss_history.append((total_loss, lattice_loss, coord_loss, type_loss, prop_loss))
            epoch_total_loss += total_loss.item()
            epoch_lattice_loss += lattice_loss.item()
            epoch_coord_loss += coord_loss.item()
            epoch_type_loss += type_loss.item()
            # epoch_prop_loss += prop_loss.item()
            epoch_sg_loss += sg_loss.item()
            epoch_wyckoff_label_loss += wyckoff_label_loss.item()
            epoch_wyckoff_param_loss += wyckoff_param_loss.item()

            num_batches += 1

        avg_total_loss = epoch_total_loss / num_batches

        loss_history.append((
            avg_total_loss,
            epoch_lattice_loss / num_batches,
            epoch_coord_loss / num_batches,
            epoch_type_loss / num_batches,
            epoch_sg_loss / num_batches,
            epoch_wyckoff_label_loss / num_batches,
            epoch_wyckoff_param_loss / num_batches
            # epoch_prop_loss / num_batches
        ))

        print(f"Epoch {epoch}/{epochs}, total_loss={avg_total_loss:.4f}")
        print(f"Epoch {epoch}/{epochs}, Lattice Loss: {epoch_lattice_loss / num_batches:.4f} | "
              f"Coords Loss: {epoch_coord_loss / num_batches:.4f} | Type Loss: {epoch_type_loss / num_batches:.4f}")
              # f"Prop Loss: {epoch_prop_loss / num_batches:.4f}")
        print(f"[Epoch {epoch}] sg_loss={epoch_sg_loss / num_batches:.4f}, wyckoff_label_loss={epoch_wyckoff_label_loss / num_batches:.4f}, wyckoff_param_loss={epoch_wyckoff_param_loss / num_batches:.4f}")

        if epoch % 1 == 0:
            # === Summarize the predicted type statistics for each epoch and check the atomic coordinates ===
            epoch_pred_types = []
            epoch_true_types = []

            os.makedirs("type_distribution", exist_ok=True)
            os.makedirs("epoch_vis", exist_ok=True)

            model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):
                    types_label = batch_data["types_label"].to(device)
                    node_fea = batch_data["node_fea"].to(device)
                    edge_index = batch_data["edge_index"].to(device)
                    edge_fea = batch_data["edge_fea"].to(device)
                    crystal_atom_idx = batch_data["crystal_atom_idx"].to(device)
                    extra_fea = batch_data["extra_fea"].to(device)
                    coords_label = batch_data["coords_label"].to(device)

                    sg_id = batch_data["sg_id"].to(device)                             # [B]
                    wyckoff_labels = batch_data["wyckoff_labels"]                     # list of list
                    wyckoff_params = batch_data["wyckoff_params"] 

                    latent = model.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)
                    sg_logits = model.decoder.sg_fc(latent)

                    _, _, coords_pred, types_logits_pred, pred_wp_logits, pred_wp_params, prop_out = model(
                        node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
                    )


                    # --- 3. Print space group predictions ---
                    sg_pred = torch.argmax(sg_logits, dim=-1) + 1   # Predicted space group numbers
                    # print(f"[Epoch {epoch}] Sample {i} SG: GT = {sg_id.tolist()}, Pred = {sg_pred.tolist()}")

                    # --- 4. Print the predictions vs. ground truth for the first few Wyckoff labels ---
                    if isinstance(wyckoff_labels[0][0], str):  # Ensure they are strings
                        label2index_inv = {v: k for k, v in label2index.items()}
                        wp_gt = []
                        for wp_list in wyckoff_labels:
                            wp_gt.append([label2index.get(lbl, 0) for lbl in wp_list])  # [B, max_atoms]
                        wp_gt = torch.tensor(wp_gt, dtype=torch.long).to(device)
                    else:
                        wp_gt = torch.tensor(wyckoff_labels, dtype=torch.long).to(device)

                    wp_pred = torch.argmax(pred_wp_logits, dim=-1)  # [B, max_atoms]

                    # for b in range(min(2, wp_gt.size(0))):  # Print the first 2 samples
                    #     gt_labels = [label2index_inv.get(i.item(), '0') for i in wp_gt[b]]
                    #     pred_labels = [label2index_inv.get(i.item(), '0') for i in wp_pred[b]]
                        # print(f"[Epoch {epoch}] Sample {b} Wyckoff Labels:")
                        # print("    GT   :", gt_labels)
                        # print("    Pred :", pred_labels)

                    # --- 5. Print Wyckoff parameters ---
                    wp_gt_param_tensor = []
                    for param_list in wyckoff_params:
                        padded = []
                        for p in param_list:
                            p = p[:6] + [0.0] * (6 - len(p))
                            padded.append(p)
                        wp_gt_param_tensor.append(torch.tensor(padded))
                    wp_gt_param_tensor = torch.stack(wp_gt_param_tensor).to(device)

                    # for b in range(min(2, wp_gt_param_tensor.shape[0])):
                        # print(f"[Epoch {epoch}] Sample {b} Wyckoff Params:")
                        # print("    GT   :", wp_gt_param_tensor[b, :4])
                        # print("    Pred :", pred_wp_params[b, :4].detach().cpu())

                    # === Collect type distribution statistics (entire batch)　===
                    pred_types_2d = types_logits_pred.view(-1, types_logits_pred.size(-1))
                    pred_type = pred_types_2d.argmax(dim=-1).cpu().numpy()
                    true_type = types_label.view(-1).cpu().numpy()
                    valid_mask = true_type != 0

                    epoch_pred_types.append(pred_type[valid_mask])
                    epoch_true_types.append(true_type[valid_mask])

                    # === Visualize coordinates for the first batch only ===
                    if i == 0:
                        pred_types = types_logits_pred.argmax(dim=-1).cpu().numpy()  # [B, max_atoms]
                        coords_np = coords_pred.cpu().numpy()                         # [B, max_atoms, 3]
                        coords_true = coords_label.cpu().numpy()                      # [B, max_atoms, 3]
                        true_types = types_label.cpu().numpy()                        # [B, max_atoms]

                        for b in range(pred_types.shape[0]):
                            pred_t = pred_types[b]
                            coords_b = coords_np[b]
                            true_t = true_types[b]
                            coords_true_b = coords_true[b]

                            valid_mask_pred = pred_t != 0
                            valid_mask_true = true_t != 0

                            coords_valid_pred = coords_b[valid_mask_pred]
                            coords_valid_true = coords_true_b[valid_mask_true]

                            # print(f"[Epoch {epoch}] Sample {b}: Predicted atom count = {valid_mask_pred.sum()}")
                            # print(pred_t)
                            # print(f"[Epoch {epoch}] Sample {b}: Ground truth atom count = {valid_mask_true.sum()}")
                            # print(true_t)

                            # print(f"[Epoch {epoch}] Sample {b}: Unique predicted types = {np.unique(pred_t[valid_mask_pred])}")
                            # print(f"[Epoch {epoch}] Sample {b}: Unique ground truth types = {np.unique(true_t[valid_mask_true])}")
                            
                            if coords_valid_pred.shape[0] > 1:
                                dists_pred = np.linalg.norm(
                                    coords_valid_pred[:, None, :] - coords_valid_pred[None, :, :], axis=-1)
                                np.fill_diagonal(dists_pred, np.inf)
                                min_d_pred = np.min(dists_pred)
                                # print(f"[Epoch {epoch}] Sample {b}: Predicted min_dist = {min_d_pred:.3f}")
                                # print(f"[Epoch {epoch}] Sample {b} pred coords (first 3 atoms):\n", coords_valid_pred[:3])

                            if coords_valid_true.shape[0] > 1:
                                dists_true = np.linalg.norm(
                                    coords_valid_true[:, None, :] - coords_valid_true[None, :, :], axis=-1)
                                np.fill_diagonal(dists_true, np.inf)
                                min_d_true = np.min(dists_true)
                                # print(f"[Epoch {epoch}] Sample {b}: Ground truth min_dist = {min_d_true:.3f}")
                                # print(f"[Epoch {epoch}] Sample {b} true coords (first 3 atoms):\n", coords_valid_true[:3])

            atom_count_pred = [np.sum(p != 0) for p in pred_types]
            atom_count_true = [np.sum(t != 0) for t in true_types]
            errors = np.array(atom_count_pred) - np.array(atom_count_true)
            print(f"[Epoch {epoch}] Avg atom count error: {errors.mean():.2f}")


            # === Summarize and Visualize ===
            pred_type_all = np.concatenate(epoch_pred_types)
            true_type_all = np.concatenate(epoch_true_types)
            pred_counts = np.bincount(pred_type_all, minlength=types_logits_pred.size(-1))
            true_counts = np.bincount(true_type_all, minlength=types_logits_pred.size(-1))
            epoch_type_stats.append((pred_counts, true_counts))

            x = np.arange(len(pred_counts))
            plt.figure(figsize=(12, 5))
            plt.bar(x - 0.2, true_counts, width=0.4, label="True", color='skyblue')
            plt.bar(x + 0.2, pred_counts, width=0.4, label="Predicted", color='salmon')
            plt.xlabel("Atom Type Index")
            plt.ylabel("Count")
            plt.title(f"Epoch {epoch}: Predicted vs True Atom Type Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"type_distribution/epoch_{epoch:03d}_type_distribution.png", dpi=300)
            # plt.close()

            plt.figure(figsize=(10, 4))
            plt.hist(errors, bins=np.arange(-MAX_ATOMS, MAX_ATOMS+2, 1), color='orange', edgecolor='black')
            plt.title(f"Epoch {epoch}: Atom Count Error Histogram (Predicted - Ground Truth)")
            plt.xlabel("Atom Count Error")
            plt.ylabel("Number of Samples")
            plt.tight_layout()
            plt.savefig(f"epoch_vis/epoch_{epoch:03d}_atom_count_error_hist.png", dpi=300)
            # plt.close()

            model.train()
            

        if avg_total_loss < best_val_loss:
            best_val_loss = avg_total_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(">> New best model found. Saving.")
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            print(f">> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(">> Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, loss_history, class_weights


class StructureDecoder(nn.Module):
    """
    Decoder that generates crystal structures with symmetry constraints (Wyckoff positions).
    
    Given a latent vector (and optionally a target space group), it predicts:
      - Lattice parameters (a, b, c, alpha, beta, gamma)
      - A space group number (1 to 230) if not provided as input
      - A set of Wyckoff positions (e.g., ['4a', '8c', ...]) for atomic sites
      - An atomic species for each Wyckoff site
      - Free fractional coordinate parameters for each Wyckoff site (if required)
    
    It then uses the PyXtal library to expand these Wyckoff positions into full atomic fractional coordinates consistent with the symmetry, and outputs:
      - `lattice`: tensor of shape [B,6] with (a,b,c,alpha,beta,gamma) in Angstrom/degree
      - `lattice_norm`: tensor of shape [B,6] with normalized lattice parameters in [0,1]
      - `coords`: tensor of shape [B, max_atoms, 3] of fractional coordinates for all atoms (padded with 0 if fewer than max_atoms)
      - `types_logits`: tensor of shape [B, max_atoms, num_atom_types] with logits for atomic types (0 index = empty site). 
    
    Usage:
      decoder = StructureDecoder(latent_dim=..., max_atoms=20, num_atom_types=len(z2index)+1, 
                                 z2index=z2index, allowed_elements=ALLOWED_ELEMENTS)
      lattice, lattice_norm, coords, types_logits = decoder(z, sg_id=None)
    
    If `sg_id` is provided, the decoder will use that space group for generation; if None, it will sample a space group (uniformly by default).
    
    Note: This decoder is primarily for inference/generation. Training it end-to-end on Wyckoff parameters is not yet supported due to lack of supervised data. During generation, it ensures the total number of atoms is between 2 and max_atoms and the number of distinct elements ≤ 6.
    """
    def __init__(self, latent_dim: int, max_atoms: int, num_atom_types: int,
                 z2index: dict, allowed_elements: list = None,
                 num_wp_labels: int = 60,
                 max_wp_param_dim: int = 6,
                 a_min=3.0, a_max=18.0, alpha_min=45.0, alpha_max=135.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types  # includes index 0 for "empty"
        # Lattice parameter ranges
        self.a_min = a_min
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Mappings for atomic types
        self.z2index = z2index  # mapping atomic number -> index used in model
        # Create inverse mapping index -> atomic number (index 0 is reserved for no-atom)
        self.index2z = {idx: Z for Z, idx in z2index.items()}
        
        # If allowed_elements list provided, filter our index2z to only those; else use all from z2index
        if allowed_elements is not None:
            self.allowed_elements = allowed_elements
            # Ensure allowed list contains symbols; convert to set for quick lookup
            self.allowed_set = set(allowed_elements)
        else:
            # If not provided, allow all elements present in z2index
            self.allowed_elements = [Element.from_Z(Z).symbol for Z in self.z2index.keys()]
            self.allowed_set = set(self.allowed_elements)
        
        # Hidden dimension for decoder layers
        hidden_dim = 256
        
        # Fully connected layers for lattice parameters prediction
        self.lattice_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # outputs raw lattice parameters (unscaled)
        )
        # Classification layer for space group prediction (1-230)
        self.sg_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 230)  # logits for each space group
        )

        # Predict coordinate [max_atoms, 3]
        self.coord_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms * 3)
        )

        # We keep a type prediction head to utilize the model's learned composition distribution
        # This outputs logits for each potential atom position (max_atoms) and type (including 0 as empty)
        self.type_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms * num_atom_types)
        )

        # New: Number of Wyckoff classification labels (typically ranges from 30 to 60)
        self.num_wp_labels = num_wp_labels
        self.max_wp_param_dim = max_wp_param_dim

        # Wyckoff label classification head (per site)
        self.wyckoff_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_atoms * self.num_wp_labels)  # 每个 site 一个分类
        )

        # Wyckoff free parameter prediction head (per site)
        self.wp_param_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.max_atoms * self.max_wp_param_dim)
        )
    
    def _map_range(self, val, min_val, max_val):
        # Map a tensor from [-1,1] to [min_val, max_val]
        return 0.5 * (val + 1.0) * (max_val - min_val) + min_val
    
    
    
    def _adjust_lattice_to_symmetry(self, a, b, c, alpha, beta, gamma, sg):
        """
        Automatically adjust the lattice parameters (a, b, c, α, β, γ) based on the space group to satisfy symmetry constraints, and automatically correct them when invalid values are detected.
        """
        # Preliminary crystal system determination
        if 195 <= sg <= 230:
            # Cubic: a = b = c, α = β = γ = 90
            a = b = c = (a + b + c) / 3.0
            alpha = beta = gamma = 90.0
            lat_type = 'cubic'

        elif 168 <= sg <= 194:
            # Hexagonal: a = b ≠ c, α = β = 90, γ = 120
            a = b = (a + b) / 2.0
            alpha = beta = 90.0
            gamma = 120.0
            lat_type = 'hexagonal'

        elif 143 <= sg <= 167:
            # Trigonal (可为 rhombohedral 或 hexagonal)
            R_centered = sg in [146, 148] or (155 <= sg <= 167)
            if R_centered:
                # Rhombohedral: a = b = c, α = β = γ ≠ 90
                a = b = c = (a + b + c) / 3.0
                ang = (alpha + beta + gamma) / 3.0
                alpha = beta = gamma = ang
                lat_type = 'trigonal'
            else:
                # Hexagonal setting
                a = b = (a + b) / 2.0
                alpha = beta = 90.0
                gamma = 120.0
                lat_type = 'hexagonal'

        elif 75 <= sg <= 142:
            # Tetragonal: a = b ≠ c, α = β = γ = 90
            a = b = (a + b) / 2.0
            alpha = beta = gamma = 90.0
            lat_type = 'tetragonal'

        elif 16 <= sg <= 74:
            # Orthorhombic: α = β = γ = 90
            alpha = beta = gamma = 90.0
            lat_type = 'orthorhombic'

        elif 3 <= sg <= 15:
            # Monoclinic: α = γ = 90, β ≠ 90
            alpha = gamma = 90.0
            lat_type = 'monoclinic'

        else:
            # Triclinic: no constraint
            lat_type = 'triclinic'

        # === Verify whether the parameters are valid (i.e., can form a valid lattice) ===
        try:
            _ = PyxLattice.from_para(a, b, c, alpha, beta, gamma, ltype=lat_type)
        except Exception as e:
            print(f"[Warning] Invalid lattice for SG={sg}, lat_type={lat_type}")
            print(f"  Params: a={a:.2f}, b={b:.2f}, c={c:.2f}, α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}")
            print("  Error:", e)

            # Project into a more valid region based on the crystal system
            if lat_type == 'trigonal':
                a = b = c = max(a, b, c)
                alpha = beta = gamma = 120.0
            elif lat_type == 'hexagonal':
                a = b = (a + b) / 2.0
                alpha = beta = 90.0
                gamma = 120.0
            elif lat_type == 'cubic':
                a = b = c = (a + b + c) / 3.0
                alpha = beta = gamma = 90.0
            elif lat_type == 'monoclinic':
                alpha = gamma = 90.0
                beta = 110.0
            elif lat_type == 'orthorhombic':
                alpha = beta = gamma = 90.0
            elif lat_type == 'tetragonal':
                a = b = (a + b) / 2.0
                alpha = beta = gamma = 90.0
            elif lat_type == 'triclinic':
                alpha = 80.0
                beta = 85.0
                gamma = 75.0

            print(f"[Fix] Adjusted to: a={a:.2f}, b={b:.2f}, c={c:.2f}, α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}")

        return a, b, c, alpha, beta, gamma, lat_type

    def generate_structure_with_symmetry(self, z, lattice, lattice_norm, types_logits, sg_id=None):
        """
        Generate crystal structures using PyXtal (used during inference/generation phase).
        """

        device = lattice.device  # FIX: 统一设备
        rng = random.Random()
        rng.seed(int(torch.randint(0, 2**31-1, (1,), device=device).item()))
        # 把 random.choice(...) 换成 rng.choice(...)
        batch_size = z.size(0)
        # --- helpers ----------------------------------------------------------
        def _allowed_symbols():
            # 兼容 allowed_set 为 Z 或 symbol 两种形式
            if all(isinstance(x, int) for x in self.allowed_set):
                return {Element.from_Z(z).symbol for z in self.allowed_set}
            return set(self.allowed_set)
        # ----------------------------------------------------------------------

        # Prepare output tensors (在正确 device 上)
        lattice_out = torch.zeros((batch_size, 6), dtype=torch.float, device=device)  # FIX
        coords_out = torch.zeros((batch_size, self.max_atoms, 3), dtype=torch.float, device=device)  # FIX
        types_logits_out = torch.zeros((batch_size, self.max_atoms, self.num_atom_types), dtype=torch.float, device=device)  # FIX

        for idx in range(batch_size):
            # Extract lattice parameters for this sample (as python floats for PyXtal)
            a, b, c, alpha, beta, gamma = lattice[idx].tolist()

            # Determine space group
            if sg_id is not None:
                sg = int(sg_id)
            else:
                if self.training:
                    sg_probs = F.softmax(self.sg_fc(z[idx:idx+1]), dim=1)  # [1,230]
                    sg = int(torch.argmax(sg_probs, dim=1).item() + 1)
                else:
                    sg = int(torch.randint(1, 231, (1,), device=device).item())
            sg = max(1, min(230, sg))

            # Adjust lattice parameters to fit the symmetry of the chosen space group
            a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj, lat_type = \
                self._adjust_lattice_to_symmetry(a, b, c, alpha, beta, gamma, sg)

            # Determine total number of atoms to place (between 2 and max_atoms)
                        # Determine total number of atoms to place (between 2 and max_atoms)
            total_atoms = self.max_atoms  # default fallback

            # 拿到 types_raw_all 供元素预测使用
            types_raw_all = self.type_fc(z[idx:idx+1]).view(1, self.max_atoms, self.num_atom_types)

            if not self.training:
                pred_type_indices = torch.argmax(types_raw_all, dim=-1).view(-1)  # [max_atoms]
                count = int((pred_type_indices != 0).sum().item())
                total_atoms = max(2, min(self.max_atoms, count))

            total_atoms = max(2, min(self.max_atoms, total_atoms))

            # Choose a set of Wyckoff positions whose multiplicities sum to total_atoms
            group = Group(sg)
            wyckoff_sites = []
            site_count = 0
            remaining = total_atoms
            wp_list = list(group)

            while remaining > 0 and site_count < 6:
                valid_wps = [wp for wp in wp_list if wp.multiplicity <= remaining]
                if len(valid_wps) == 0:
                    break
                wp_choice = rng.choice(valid_wps)
                wyckoff_sites.append(wp_choice)
                site_count += 1
                remaining -= wp_choice.multiplicity
                if remaining == 0:
                    break

            if remaining != 0:
                wyckoff_sites.clear()
                for wp in wp_list:
                    if wp.multiplicity <= total_atoms:
                        wyckoff_sites.append(wp)
                        break

            # ====== Species assignment without composition prior ======
            predicted_species_set = []

            if not self.training:
                unique_type_indices = pred_type_indices[pred_type_indices != 0].unique().tolist()
                for ti in unique_type_indices:
                    if ti in self.index2z:
                        predicted_Z = self.index2z[ti]
                        sym = Element.from_Z(predicted_Z).symbol
                        if sym in _allowed_symbols():
                            predicted_species_set.append(predicted_Z)

            predicted_species_set = predicted_species_set[:6]
            species_for_sites = []

            if len(predicted_species_set) > 0:
                for i, wp in enumerate(wyckoff_sites):
                    if i < len(predicted_species_set):
                        Z = predicted_species_set[i]
                        sym = Element.from_Z(Z).symbol
                        if sym not in _allowed_symbols():
                            sym = rng.choice(list(_allowed_symbols()))
                        species_for_sites.append(sym)
                    else:
                        available = [el for el in self.allowed_elements if el not in species_for_sites]
                        if len(available) == 0:
                            available = self.allowed_elements
                        sym = rng.choice(available)
                        species_for_sites.append(sym)
            else:
                used = []
                for i, wp in enumerate(wyckoff_sites):
                    available = [el for el in self.allowed_elements if el not in used]
                    if len(available) == 0:
                        available = self.allowed_elements
                    sym = rng.choice(available)
                    species_for_sites.append(sym)
                    if sym not in used and len(used) < 6:
                        used.append(sym)
            # ====== End of species assignment ======

            # Lattice sanity (optional)
            try:
                _ = PyxLattice.from_para(a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj, ltype=lat_type)
            except Exception as e:
                print(f"[Warning] Invalid lattice for SG={sg}, lat_type={lat_type}")
                print(f"  Params: a={a_adj:.2f}, b={b_adj:.2f}, c={c_adj:.2f}, α={alpha_adj:.1f}, β={beta_adj:.1f}, γ={gamma_adj:.1f}")
                print(f"  Error: {e}")

            # List of Wyckoff position labels (e.g., '4a') for each site
            wp_labels = [f"{wp.multiplicity}{wp.letter}" for wp in wyckoff_sites]

            # FIX: x_array = 6个物理晶格参数 + 各位点 DOF（绝不按晶系减少前6项）
            cell_params = [a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj]  # FIX: 固定 6 项
            free_params = []
            for wp in wyckoff_sites:
                dof = wp.get_dof()
                if dof > 0:
                    free_params.extend([random.random() for _ in range(dof)])
            x_array = cell_params + free_params  # FIX

            # Build with PyXtal
            xtal = pyxtal()
            try:
                xtal.from_spg_wps_rep(sg, wp_labels, x_array, species_for_sites)
            except Exception as e:
                print(f"[ERROR] Initial build failed: {e}")
                print(f"  SG: {sg}, WP: {wp_labels}, Species: {species_for_sites}")
    
                try:
                    group = Group(sg)
                    gp = group[0]
                    reps = max(1, min(12, math.ceil(total_atoms / gp.multiplicity)))
                    wyckoff_sites = [gp for _ in range(reps)]
                    wp_labels = [f"{gp.multiplicity}{gp.letter}" for _ in wyckoff_sites]
                    species_for_sites = [rng.choice(self.allowed_elements) for _ in wyckoff_sites]
                    # 重建 x_array
                    free_params = []
                    for _ in wyckoff_sites:
                        dof = gp.get_dof()
                        if dof > 0:
                            free_params.extend([random.random() for _ in range(dof)])
                    x_array = cell_params + free_params
                    xtal = pyxtal()
                    xtal.from_spg_wps_rep(sg, wp_labels, x_array, species_for_sites)
                except Exception as e2:
                    print(f"[Fallback] General Wyckoff build failed: {e2}")
                
                    try:
                        sg = 1
                        group = Group(sg)
                        gp = group[0]
                        reps = max(1, min(self.max_atoms, total_atoms))
                        wyckoff_sites = [gp for _ in range(reps)]
                        wp_labels = [f"{gp.multiplicity}{gp.letter}" for _ in wyckoff_sites]
                        species_for_sites = [rng.choice(self.allowed_elements) for _ in wyckoff_sites]
                        free_params = []
                        for _ in wyckoff_sites:
                            dof = gp.get_dof()
                            if dof > 0:
                                free_params.extend([random.random() for _ in range(dof)])
                        x_array = cell_params + free_params
                        xtal = pyxtal()
                        xtal.from_spg_wps_rep(sg, wp_labels, x_array, species_for_sites)
                    except Exception as e3:
                        print(f"[Final Fallback] SG=1 build also failed.")
                        raise e3

            # Convert to pymatgen & write outputs
            print("DBG species_for_sites:", species_for_sites)
            pmg_struct = xtal.to_pymatgen()
            print("DBG GEN COMP:", dict(Counter([sp.symbol for sp in pmg_struct.species])))
            sites = pmg_struct.sites
            num_atoms = min(len(sites), self.max_atoms)

            # Fill lattice_out
            lattice_out[idx, :] = torch.tensor([a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj],
                                            dtype=torch.float, device=device)

            # Fill coords_out & types_logits_out
            for j in range(num_atoms):
                coords_out[idx, j, :] = torch.tensor(list(sites[j].frac_coords), dtype=torch.float, device=device)
                sym = sites[j].specie.symbol
                try:
                    Z = Element(sym).Z
                except Exception:
                    Z = Element(sym.capitalize()).Z

                type_index = self.z2index.get(Z, 0)  # FIX: 生成阶段不要改映射，未知映射到 0
                if 0 <= type_index < self.num_atom_types:
                    types_logits_out[idx, j, type_index] = torch.tensor(10.0, device=device)

        return lattice_out, lattice_norm, coords_out, types_logits_out

    
    def forward(
            self, 
            z: torch.Tensor, 
            mode: str = "train", 
            sg_id: int = None
        ):
        """
        Generate a crystal structure from latent vector z (optionally specifying a space group).
        
        Args:
            z (torch.Tensor): latent vector of shape [B, latent_dim].
            sg_id (int, optional): if provided, use this space group number for generation (1-230).
        
        Returns:
            lattice (torch.Tensor): [B, 6] tensor of (a, b, c, alpha, beta, gamma).
            lattice_norm (torch.Tensor): [B, 6] tensor of normalized lattice parameters (0-1 range).
            coords (torch.Tensor): [B, max_atoms, 3] tensor of fractional coordinates for all atoms.
            types_logits (torch.Tensor): [B, max_atoms, num_atom_types] tensor of logits for atomic types (including 0 for empty).
        """
        batch_size = z.size(0)

        # 1. Lattice parameter prediction (raw)
        raw_lattice = self.lattice_fc(z)  # [B,6]
        raw_a, raw_b, raw_c, raw_alpha, raw_beta, raw_gamma = torch.chunk(raw_lattice, chunks=6, dim=1)
        # Map raw values to [-1,1] via tanh, then to physical ranges [min, max]
        a = self._map_range(torch.tanh(raw_a), self.a_min, self.a_max)
        b = self._map_range(torch.tanh(raw_b), self.a_min, self.a_max)
        c = self._map_range(torch.tanh(raw_c), self.a_min, self.a_max)
        alpha = self._map_range(torch.tanh(raw_alpha), self.alpha_min, self.alpha_max)
        beta  = self._map_range(torch.tanh(raw_beta),  self.alpha_min, self.alpha_max)
        gamma = self._map_range(torch.tanh(raw_gamma), self.alpha_min, self.alpha_max)
        
        # Concatenate for normalized output (0-1 scaling)
        a_01     = (a - self.a_min) / (self.a_max - self.a_min)
        b_01     = (b - self.a_min) / (self.a_max - self.a_min)
        c_01     = (c - self.a_min) / (self.a_max - self.a_min)
        alpha_01 = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
        beta_01  = (beta - self.alpha_min) / (self.alpha_max - self.alpha_min)
        gamma_01 = (gamma - self.alpha_min) / (self.alpha_max - self.alpha_min)
        
        # Reconstruct the physical values of the lattice
        lattice = torch.cat([a, b, c, alpha, beta, gamma], dim=1)  # shape [B,6]

        # lattice_norm is also in [0, 1] (aligned with ‘lattice_batch’ in train_decoder)
        lattice_norm = torch.cat([a_01, b_01, c_01, alpha_01, beta_01, gamma_01], dim=1)  # [B,6]

        # ============= 2) coords (fractional) =============
        coords_raw = self.coord_fc(z)  # [B, max_atoms*3]
        coords_raw = coords_raw.view(batch_size, self.max_atoms, 3)
        # Use tanh instead of sigmoid to avoid collapse and map to [0, 1]
        coords_tanh = torch.tanh(coords_raw)  # [B, max_atoms, 3]
        coords = 0.5 * (coords_tanh + 1.0) # Map to [0, 1]

        # Add slight perturbation during training to increase diversity
        if self.training:
            noise = torch.randn_like(coords) * 0.01  # Standard deviation is adjustable
            coords = coords + noise
            coords = torch.clamp(coords, 0.0, 1.0)  # Ensure values remain within valid range

        # ============= 3) Atomic type logits =============
        types_raw = self.type_fc(z).view(batch_size, self.max_atoms, self.num_atom_types)
        # types_raw is used directly as logits
        types_logits = types_raw

        if mode == "train":
            # === Wyckoff label logits ===
            pred_wp_logits = self.wyckoff_fc(z)  # [B, max_atoms * num_wp_labels]
            pred_wp_logits = pred_wp_logits.view(batch_size, self.max_atoms, self.num_wp_labels)

            # === Wyckoff free parameters ===
            pred_wp_params = self.wp_param_fc(z)  # [B, max_atoms * param_dim]
            pred_wp_params = pred_wp_params.view(batch_size, self.max_atoms, self.max_wp_param_dim)

            return lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params
                
        elif mode == "generate":
            # Follow the original PyXtal generation pipeline
            return self.generate_structure_with_symmetry(
                z, lattice, lattice_norm, types_logits, sg_id
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        

class PropertyPredictor(nn.Module):
    """
    Auxiliary network to predict material properties (conditions) from latent vector.
    Outputs a vector of length cond_dim (e.g., [melting_point_log, density, formation_energy] normalized).
    """
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, cond_dim)
    def forward(self, z: torch.Tensor):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        # Output is direct (we will interpret this as normalized property predictions)
        out = self.fc_out(x)
        return out


def evaluate_encoder_decoder(
    model: EncoderDecoderModel,
    dataset: StructureDataset,
    label2index,
    batch_size: int = 32,
    device=torch.device("cpu"), 
    class_weights=None,
    num_atom_types=85,
    lattice_w=10.0,
    coord_w=5.0,
    type_w=2.0,
    prop_w=5.0
):
    model.eval()
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    lattice_losses, coord_losses, type_losses, prop_losses, total_losses = [], [], [], [], []
    lattice_stds, coord_diversities, type_entropies = [], [], []
    wyckoff_sg_losses, wyckoff_label_losses, wyckoff_param_losses = [], [], []
    
    all_pred_types = []
    all_true_types = []


    with torch.no_grad():
        for batch_data in dataloader:
            node_fea = batch_data["node_fea"].to(device)               # [N_atoms]
            edge_index = batch_data["edge_index"].to(device)           # [2, E]
            edge_fea = batch_data["edge_fea"].to(device)               # [E, edge_fea_dim]
            crystal_atom_idx = batch_data["crystal_atom_idx"].to(device) # [N_atoms]
            extra_fea = batch_data["extra_fea"].to(device)             # [1, extra_dim]
            cond = batch_data["cond"].to(device)                       # [cond_dim]
            lattice_label = batch_data["lattice_label"].to(device)     # [6]
            coords_label = batch_data["coords_label"].to(device)       # [max_atoms,3]
            types_label = batch_data["types_label"].to(device)         # [max_atoms]

            # forward
            _, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params, prop_out = model(
                node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
            )

            # 1) lattice loss
            lattice_loss = F.mse_loss(lattice_norm, lattice_label, reduction='mean')
            lattice_losses.append(lattice_loss.item())

            lattice_std = torch.std(lattice_norm, dim=1).mean().item()
            lattice_stds.append(lattice_std)


            # 2) coords loss
            mask = (types_label > 0).unsqueeze(-1).float()  # [B, max_atoms, 1]
            coord_diff = (coords - coords_label) * mask
            coord_loss = (coord_diff**2).sum() / (mask.sum() * 3.0 + 1e-8)
            coord_losses.append(coord_loss.item())
            dist_loss = atom_pairwise_dist_penalty(coords, mask)

            # coord diversity
            def coord_pairwise_diversity(coords, mask):
                """
                coords: [B, N, 3]
                mask: [B, N, 1]
                """
                B, N, _ = coords.shape
                loss = 0.0
                for b in range(B):
                    valid = mask[b].squeeze(-1) > 0  # shape [N]
                    c = coords[b][valid]             # [N_valid, 3]
                    if c.size(0) <= 1: continue
                    diff = c.unsqueeze(0) - c.unsqueeze(1)  # [N, N, 3]
                    dist = (diff**2).sum(-1) + 1e-8        
                    dist = torch.sqrt(dist)                # [N, N]
                    triu_mask = torch.triu(torch.ones_like(dist), diagonal=1)
                    pairwise_dists = dist[triu_mask == 1]
                    loss += -pairwise_dists.mean()
                return loss / B

            coord_diversity_loss = coord_pairwise_diversity(coords, mask)
            coord_diversities.append(coord_diversity_loss.item())

            # 3) type loss
            type_loss = 0.0
            type_entropy_bonus = 0.0
            if types_logits is not None:
                pred_types_2d = types_logits.view(-1, types_logits.size(-1))
                
                # types_logits: [B, max_atoms, num_atom_types]
                # types_label: [B, max_atoms]
                valid_mask = types_label.view(-1) > -1
                # type_loss = F.cross_entropy(
                #     pred_types_2d[valid_mask], 
                #     types_label.view(-1)[valid_mask], 
                #     weight=class_weights
                # )
                class_weights[0] = 0.05

                type_loss = focal_loss(
                    logits=pred_types_2d[valid_mask],
                    targets=types_label.view(-1)[valid_mask],
                    alpha=class_weights.to(device), 
                    gamma=1.5,                  
                    # ignore_index=0               # ignore padding type
                )

                type_losses.append(type_loss.item())
                # types_logits[..., 0] = -1e9 
                
                type_count_loss = nonzero_type_penalty(types_logits, target_min=4)
                true_atom_counts = (types_label > 0).sum(dim=1).float()  # [B]
                pred_types = types_logits.argmax(dim=-1)
                pred_atom_counts = (pred_types != 0).sum(dim=1).float()
                count_loss = F.mse_loss(pred_atom_counts, true_atom_counts)

                pred_type_probs = F.softmax(torch.clamp(pred_types_2d[valid_mask], -30, 30), dim=-1)
                entropy = - (pred_type_probs * torch.log(pred_type_probs + 1e-8)).sum(dim=-1)
                mean_entropy = entropy.mean().item()
                type_entropies.append(entropy.mean().item())
                type_entropy_bonus = mean_entropy

                # Save predicted/ground-truth types
                pred_type = pred_type_probs.argmax(dim=-1).cpu().numpy()
                true_type = types_label.view(-1)[valid_mask].cpu().numpy()
                all_pred_types.extend(pred_type)
                all_true_types.extend(true_type)

            # 4) property loss
            # prop_loss = 0.0
            # if prop_out is not None:
            #     prop_loss = F.mse_loss(prop_out, cond, reduction='mean')
            #     prop_losses.append(prop_loss.item())


            try:
                latent = model.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)
                sg_logits = model.decoder.sg_fc(latent)
                sg_loss, wyckoff_label_loss, wyckoff_param_loss = compute_wyckoff_losses(
                    sg_logits, pred_wp_logits, pred_wp_params,
                    batch_data, label2index, device
                )
                wyckoff_sg_losses.append(sg_loss.item())
                wyckoff_label_losses.append(wyckoff_label_loss.item())
                wyckoff_param_losses.append(wyckoff_param_loss.item())
            except Exception as e:
                print(f"[Warning][Eval] Wyckoff loss computation failed: {e}")


            # === Lattice diversity penalty ===
            def pairwise_var_loss(x):
                diff = x.unsqueeze(1) - x.unsqueeze(0)
                dist = (diff ** 2).sum(-1)
                return -dist.mean()
            diversity_penalty = - 0.01 * pairwise_var_loss(lattice_norm)

            # === Total loss (same as training) ===
            total_loss = (
                lattice_w * lattice_loss +
                diversity_penalty +
                coord_w * coord_loss +
                0.2 * coord_diversity_loss +  # same sign as training
                2.0 * dist_loss + 
                type_w * type_loss +
                -0.1 * type_entropy_bonus +
                0.5 * type_count_loss +  
                1.0 * count_loss +
                1.0 * sg_loss +
                1.0 * wyckoff_label_loss + 
                0.5 * wyckoff_param_loss
                # prop_w * prop_loss
            )
            total_losses.append(total_loss.item())
        


        # === Visualize the prediction results of the first 30 samples ===
        print("\n[Evaluation] Sample-wise Predictions (First 30):")
        sample_count = 0
        for batch_data in dataloader:
            types_label = batch_data["types_label"].to(device)
            coords_label = batch_data["coords_label"].to(device)
            lattice_label = batch_data["lattice_label"].to(device)
            node_fea = batch_data["node_fea"].to(device)
            edge_index = batch_data["edge_index"].to(device)
            edge_fea = batch_data["edge_fea"].to(device)
            crystal_atom_idx = batch_data["crystal_atom_idx"].to(device)
            extra_fea = batch_data["extra_fea"].to(device)
            sg_true = batch_data["sg_id"].cpu().numpy()  # [B]
            wp_true = batch_data["wyckoff_labels"]       # [B, max_atoms], list of list of strings
            wp_params_true = batch_data["wyckoff_params"]  # [B, max_atoms, D], list of list of list


            lattice, lattice_norm, coords_pred, types_logits_pred, pred_wp_logits, pred_wp_params, prop_out = model(
                node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
            )

            lattices_pred = lattice_norm.cpu().numpy()         # shape [6], normalized lattice
            lattices_true = lattice_label.cpu().numpy()        # shape [6], normalized lattice

            
            pred_types = types_logits_pred.argmax(dim=-1).cpu().numpy()  # [B, max_atoms]
            coords_np = coords_pred.cpu().numpy()                         # [B, max_atoms, 3]
            coords_true = coords_label.cpu().numpy()                      # [B, max_atoms, 3]
            true_types = types_label.cpu().numpy()                        # [B, max_atoms]

            # 1. Predicted sg
            z = model.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)
            sg_logits = model.decoder.sg_fc(z)  # [B, 230]
            sg_pred = sg_logits.argmax(dim=-1).cpu().numpy() + 1  # 1-based index

            # 2. Predicted Wyckoff label
            wp_logits_pred = pred_wp_logits.argmax(dim=-1).cpu().numpy()  # [B, max_atoms]

            # Optional: need to reverse map label2index to index2label
            index2label = {v: k for k, v in label2index.items()}

            # 3. Predicted Wyckoff parameters
            wp_params_pred = pred_wp_params.cpu().numpy()  # [B, max_atoms, D]


            B = pred_types.shape[0]
            for b in range(B):
                if sample_count >= 30:
                    break

                pred_t = pred_types[b]
                coords_b = coords_np[b]
                true_t = true_types[b]
                coords_true_b = coords_true[b]
                lattice_pred = lattices_pred[b]
                lattice_true = lattices_true[b]

                valid_mask_pred = pred_t != 0
                valid_mask_true = true_t != 0

                coords_valid_pred = coords_b[valid_mask_pred]
                coords_valid_true = coords_true_b[valid_mask_true]

                print(f"[Eval] Sample {sample_count}: Predicted lattice = {lattice_pred}")
                print(f"[Eval] Sample {sample_count}: Ground truth lattice = {lattice_true}")

                print(f"[Eval] Sample {sample_count}: Predicted atom count = {valid_mask_pred.sum()}")
                print(pred_t)
                print(f"[Eval] Sample {sample_count}: Ground truth atom count = {valid_mask_true.sum()}")
                print(true_t)
                print(f"[Eval] Sample {sample_count}: Unique predicted types = {np.unique(pred_t[valid_mask_pred])}")
                print(f"[Eval] Sample {sample_count}: Unique ground truth types = {np.unique(true_t[valid_mask_true])}")
                
                if coords_valid_pred.shape[0] > 1:
                    dists_pred = np.linalg.norm(
                        coords_valid_pred[:, None, :] - coords_valid_pred[None, :, :], axis=-1)
                    np.fill_diagonal(dists_pred, np.inf)
                    min_d_pred = np.min(dists_pred)
                    print(f"[Eval] Sample {sample_count}: Predicted min_dist = {min_d_pred:.3f}")
                    print(f"[Eval] Sample {sample_count} pred coords (first 3 atoms):\n", coords_valid_pred[:3])

                if coords_valid_true.shape[0] > 1:
                    dists_true = np.linalg.norm(
                        coords_valid_true[:, None, :] - coords_valid_true[None, :, :], axis=-1)
                    np.fill_diagonal(dists_true, np.inf)
                    min_d_true = np.min(dists_true)
                    print(f"[Eval] Sample {sample_count}: Ground truth min_dist = {min_d_true:.3f}")
                    print(f"[Eval] Sample {sample_count} true coords (first 3 atoms):\n", coords_valid_true[:3])

                print(f"[Eval] Sample {sample_count}: Predicted SG = {sg_pred[b]}")
                print(f"[Eval] Sample {sample_count}: Ground Truth SG = {sg_true[b]}")

                true_labels_b = wp_true[b]
                pred_labels_b = [index2label.get(idx, "?") for idx in wp_logits_pred[b]]
                print(f"[Eval] Sample {sample_count}: Pred Wyckoff Labels = {pred_labels_b}")
                print(f"[Eval] Sample {sample_count}: True Wyckoff Labels = {true_labels_b}")

                print(f"[Eval] Sample {sample_count}: Pred Wyckoff Params (first 2):")
                for i in range(2):
                    print(f"    pred: {wp_params_pred[b][i]}")
                    print(f"    true: {wp_params_true[b][i]}")

                sample_count += 1
            if sample_count >= 30:
                break

    # === Print the predicted type distribution ===
    print("\n[Evaluation] Type prediction summary:")
    pred_counts = np.bincount(all_pred_types, minlength=num_atom_types)
    true_counts = np.bincount(all_true_types, minlength=num_atom_types)
    print("Predicted type counts:", pred_counts.tolist())
    print("True      type counts:", true_counts.tolist())


    cm = confusion_matrix(all_true_types, all_pred_types, labels=list(range(num_atom_types)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, cmap='Blues', square=True, cbar=True,
                xticklabels=np.arange(num_atom_types),
                yticklabels=np.arange(num_atom_types),
                linewidths=0.5, linecolor='gray')
    plt.xlabel("Predicted Type")
    plt.ylabel("True Type")
    plt.title("Normalized Confusion Matrix (Validation)")
    plt.tight_layout()

    os.makedirs("confusion_matrix", exist_ok=True)
    plt.savefig("confusion_matrix/validation_type_confusion_matrix.png", dpi=300)
    plt.close()
    print("[INFO] Confusion matrix saved to confusion_matrix/validation_type_confusion_matrix.png")

    return {
        "val_total_loss": np.mean(total_losses),
        "val_lattice_mse": np.mean(lattice_losses),
        "val_coord_mse": np.mean(coord_losses),
        "val_type_loss": np.mean(type_losses),
        # "val_prop_mse": np.mean(prop_losses),
        "val_lattice_std": np.mean(lattice_stds),
        "val_coord_diversity": np.mean(coord_diversities),
        "val_type_entropy": np.mean(type_entropies),
        "val_type_entropy": np.mean(type_entropies),
        "val_wyckoff_sg_loss": np.mean(wyckoff_sg_losses),
        "val_wyckoff_label_loss": np.mean(wyckoff_label_losses),
        "val_wyckoff_param_loss": np.mean(wyckoff_param_losses)
    }


def plot_training_losses(loss_history, figure_path):
    """
    Plots the loss curves for training history.
    Expects loss_history to be a list of tuples:
    (total_loss, lattice_loss, coord_loss, type_loss, sg_loss, wyckoff_label_loss, wyckoff_param_loss)
    """
    total_loss_list = [l[0] for l in loss_history]
    lattice_loss_list = [l[1] for l in loss_history]
    coord_loss_list = [l[2] for l in loss_history]
    type_loss_list = [l[3] for l in loss_history]
    sg_loss_list = [l[4] for l in loss_history]
    wyckoff_label_loss_list = [l[5] for l in loss_history]
    wyckoff_param_loss_list = [l[6] for l in loss_history]

    epochs = list(range(1, len(loss_history) + 1))

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # 2 rows × 4 columns

    axs[0, 0].plot(epochs, total_loss_list, label='Total Loss', color='blue')
    axs[0, 0].set_title("Total Loss")

    axs[0, 1].plot(epochs, lattice_loss_list, label='Lattice Loss', color='green')
    axs[0, 1].set_title("Lattice Loss")

    axs[0, 2].plot(epochs, coord_loss_list, label='Coord Loss', color='orange')
    axs[0, 2].set_title("Coord Loss")

    axs[0, 3].plot(epochs, type_loss_list, label='Type Loss', color='red')
    axs[0, 3].set_title("Type Loss")

    axs[1, 0].plot(epochs, sg_loss_list, label='SG Loss', color='purple')
    axs[1, 0].set_title("Space Group Loss")

    axs[1, 1].plot(epochs, wyckoff_label_loss_list, label='Wyckoff Label Loss', color='teal')
    axs[1, 1].set_title("Wyckoff Label Loss")

    axs[1, 2].plot(epochs, wyckoff_param_loss_list, label='Wyckoff Param Loss', color='brown')
    axs[1, 2].set_title("Wyckoff Param Loss")

    # Hide last unused subplot (bottom-right corner)
    axs[1, 3].axis('off')

    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    # plt.show()


class CGCNNConv_old(nn.Module):
    def __init__(self, atom_fea_len, edge_fea_len):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + edge_fea_len, 2 * atom_fea_len)

    def forward(self, atom_fea, edge_index, edge_fea, crystal_atom_idx):
        src, dst = edge_index[0], edge_index[1]
        atom_src = atom_fea[src]
        atom_dst = atom_fea[dst]

        edge_input = torch.cat([atom_src, atom_dst, edge_fea], dim=1)  # [E, 2F + E_dim]
        edge_transformed = self.fc_full(edge_input)                    # [E, 2F]
        gate, core = edge_transformed.chunk(2, dim=1)                  # [E, F] + [E, F]

        gate = torch.sigmoid(gate)
        core = torch.tanh(core)
        message = gate * core

        agg = torch.zeros_like(atom_fea)
        agg = scatter_add(message, dst, dim=0, out=agg)

        new_atom_fea = F.softplus(atom_fea + agg)
        return new_atom_fea  # residual + softplus
    

class CGCNN(nn.Module):
    def __init__(self,
                 atom_fea_dim=64,
                 edge_fea_dim=128,
                 num_targets=1,
                 num_atom_types=85,
                 depth=3, 
                 use_extra_fea=False,
                 extra_fea_dim=10):
        super().__init__()
        self.use_extra_fea = use_extra_fea   
        self.extra_fea_dim = extra_fea_dim

        self.embed = AtomEmbedding(num_atom_types, atom_fea_dim)

        self.convs = nn.ModuleList([
            CGCNNConv_old(atom_fea_dim, edge_fea_dim)
            for _ in range(depth)
        ])

        fc_input_dim = atom_fea_dim + (extra_fea_dim if use_extra_fea else 0)

        self.fc_out = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_targets)
        )

    def forward(self, node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea=None):
        x = self.embed(node_fea)  # [N, F]

        for conv in self.convs:
            x = conv(x, edge_index, edge_fea, crystal_atom_idx)

        # pooling: graph-level average
        num_graphs = crystal_atom_idx.max().item() + 1
        g_fea = scatter_mean(x, crystal_atom_idx, dim=0,
                             out=torch.zeros((num_graphs, x.size(1)), device=x.device))

        if self.use_extra_fea and extra_fea is not None:
            g_fea = torch.cat([g_fea, extra_fea], dim=1)

        # MLP
        out = self.fc_out(g_fea)
        return out
    


def compute_latent_global_stats(encoder, dataset, device, batch_size=32, collate_fn=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    encoder.eval()
    encoder.to(device)

    all_latents = []
    with torch.no_grad():
        for batch_data in dataloader:
            node_fea = batch_data["node_fea"].to(device)
            edge_index = batch_data["edge_index"].to(device)
            edge_fea = batch_data["edge_fea"].to(device)
            crystal_atom_idx = batch_data["crystal_atom_idx"].to(device)
            extra_fea = batch_data["extra_fea"].to(device)

            latent = encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)
            all_latents.append(latent.cpu())

    all_latents = torch.cat(all_latents, dim=0)
    latent_mean = all_latents.mean(dim=0)
    latent_std = all_latents.std(dim=0) + 1e-6
    return latent_mean.to(device), latent_std.to(device)


# ===== Smoothly control the sampling probabilities and loss weights =====
def warmup_cosine(epoch, warmup_epochs, max_epoch, max_val, min_val=0.01):
    if epoch <= warmup_epochs:
        return min_val + (max_val - min_val) * epoch / warmup_epochs
    else:
        return min_val + 0.5 * (max_val - min_val) * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs)))
    

def check_structure_collapse(
    lattice_norm: torch.Tensor,        # [B, 6]
    coords: torch.Tensor,              # [B, max_atoms, 3]
    types_logits: torch.Tensor,        # [B, max_atoms, num_types]
    types_mask: torch.Tensor,          # [B, max_atoms, 1]
    epoch: int,
    num_atom_types: int,
    log_dir: str = "collapse_logs",
    threshold_lattice_std: float = 1e-3,
    threshold_coord_div: float = 0.05,
    threshold_entropy: float = 0.2,
):
    os.makedirs(log_dir, exist_ok=True)

    # --- Lattice diversity ---
    lattice_std = torch.std(lattice_norm, dim=0).mean().item()
    if lattice_std < threshold_lattice_std:
        print(f"[Collapse Warning][Epoch {epoch}] Low lattice diversity: std = {lattice_std:.6f}")

    # --- Coord diversity ---
    def compute_coord_diversity(coords, mask):
        B, N, _ = coords.shape
        diversity = []
        for b in range(B):
            valid = mask[b].squeeze(-1) > 0
            c = coords[b][valid]  # [N_valid, 3]
            if c.size(0) <= 1:
                continue
            dist = torch.cdist(c, c, p=2)
            upper = torch.triu(dist, diagonal=1)
            values = upper[upper > 0]
            if values.numel() > 0:
                diversity.append(values.mean().item())
        return np.mean(diversity) if diversity else 0.0

    coord_div = compute_coord_diversity(coords, types_mask)
    if coord_div < threshold_coord_div:
        print(f"[Collapse Warning][Epoch {epoch}] Low coord diversity: {coord_div:.6f}")

    # --- Type entropy ---
    probs = F.softmax(types_logits.view(-1, types_logits.size(-1)), dim=-1)
    entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
    if entropy < threshold_entropy:
        print(f"[Collapse Warning][Epoch {epoch}] Low type entropy: {entropy:.4f}")

    # --- Type distribution plot ---
    pred_types = probs.argmax(dim=-1).cpu().numpy()
    counts = np.bincount(pred_types, minlength=num_atom_types)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=np.arange(num_atom_types), y=counts, color='salmon')
    plt.title(f"Epoch {epoch} - Predicted Type Distribution")
    plt.xlabel("Atom Type Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"epoch_{epoch:03d}_type_dist.png"), dpi=300)
    plt.close()

    # --- Save log summary ---
    with open(os.path.join(log_dir, "collapse_log.txt"), "a") as f:
        f.write(f"Epoch {epoch}\tLattice_std: {lattice_std:.4f}\tCoord_div: {coord_div:.4f}\tEntropy: {entropy:.4f}\n")

    return {
        "lattice_std": lattice_std,
        "coord_diversity": coord_div,
        "type_entropy": entropy
    }



def visualize_lattice_and_type(lattice_pred, lattice_gt, pred_types, true_types, epoch, save_dir="diff_vis"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Lattice visualization
    plt.figure(figsize=(8, 4))
    x = np.arange(6)
    plt.bar(x - 0.2, lattice_gt, width=0.4, label="GT", color='skyblue')
    plt.bar(x + 0.2, lattice_pred, width=0.4, label="Pred", color='salmon')
    plt.xticks(x, [f"L{i}" for i in range(6)])
    plt.title(f"Lattice Prediction vs Ground Truth (Epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lattice_epoch_{epoch:03d}.png")
    plt.close()

    # 2. Type distribution comparison
    max_type = max(pred_types.max(), true_types.max()) + 1
    pred_counts = np.bincount(pred_types, minlength=max_type)
    true_counts = np.bincount(true_types, minlength=max_type)
    x = np.arange(len(pred_counts))

    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, true_counts, width=0.4, label="GT", color='skyblue')
    plt.bar(x + 0.2, pred_counts, width=0.4, label="Pred", color='salmon')
    plt.xlabel("Atom Type")
    plt.ylabel("Count")
    plt.title(f"Predicted vs True Atom Types (Epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/type_epoch_{epoch:03d}.png")
    plt.close()


def evaluate_and_log_diffusion_outputs(
    epoch, 
    coords_pred, 
    coords_label, 
    lattice_pred, lattice_label, 
    types_logits, types_label, 
    visualize_fn=None
):
    B = coords_pred.size(0)
    for b in range(B):
        lattice_gt = lattice_label[b].detach().cpu().numpy()
        lattice_pr = lattice_pred[b].detach().cpu().numpy()

        pred_types = types_logits[b].argmax(dim=-1).cpu().numpy()
        true_types = types_label[b].cpu().numpy()

        coords_pr = coords_pred[b].detach().cpu().numpy()
        coords_gt = coords_label[b].cpu().numpy()

        valid_mask_pred = pred_types != 0
        valid_mask_true = true_types != 0

        coords_valid_pred = coords_pr[valid_mask_pred][:3]
        coords_valid_true = coords_gt[valid_mask_true][:3]

        print(f"[Eval] Epoch {epoch} Sample {b} lattice_pred: ", lattice_pr)
        print(f"[Eval] Epoch {epoch} Sample {b} lattice_gt: ", lattice_gt)

        print(f"[Eval] Epoch {epoch} Sample {b} pred_types: ", pred_types)
        print(f"[Eval] Epoch {epoch} Sample {b} true_types: ", true_types)

        print(f"[Eval] Epoch {epoch} Sample {b} pred coords (first 3 atoms):\n", coords_valid_pred)
        print(f"[Eval] Epoch {epoch} Sample {b} true coords (first 3 atoms):\n", coords_valid_true)

        if coords_valid_pred.shape[0] > 1:
            dists_pred = np.linalg.norm(coords_valid_pred[:, None, :] - coords_valid_pred[None, :, :], axis=-1)
            np.fill_diagonal(dists_pred, np.inf)
            min_d_pred = np.min(dists_pred)
            print(f"[Eval] Epoch {epoch} Sample {b} Predicted min_dist = {min_d_pred:.3f}")

        if coords_valid_true.shape[0] > 1:
            dists_true = np.linalg.norm(coords_valid_true[:, None, :] - coords_valid_true[None, :, :], axis=-1)
            np.fill_diagonal(dists_true, np.inf)
            min_d_true = np.min(dists_true)
            print(f"[Eval] Epoch {epoch} Sample {b} Ground truth min_dist = {min_d_true:.3f}")

        if visualize_fn and b == 0:
            visualize_fn(lattice_pr, lattice_gt, pred_types, true_types, epoch)


def is_valid_structure(struct: Structure) -> bool:
    try:
        # Check lattice parameters are finite values or not
        lengths = struct.lattice.abc  # (a, b, c)
        angles = struct.lattice.angles  # (alpha, beta, gamma)
        if not all(np.isfinite(lengths)) or not all(np.isfinite(angles)):
            return False
        if any(l <= 0 for l in lengths):
            return False
        if any(angle <= 10.0 or angle >= 170.0 for angle in angles):
            return False

        # Check distance matrix is valid or not
        dist = struct.distance_matrix
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        if not np.isfinite(min_dist) or min_dist <= 0.1:
            return False

        return True
    except Exception as e:
        print(f"[Invalid Structure] Exception while checking: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Conditional Diffusion Model for Material Generation")
    
    # Select mode
    parser.add_argument('--mode', type=str, choices=['pretrain', 'generate'], required=True,
                        help="Mode: 'train' to train models, 'generate' to sample new structures")
    
    # For pretrain
    parser.add_argument('--pretrain_data_csv', type=str, default='data_csv/data_e43V.csv',
                        help="Path to the input data CSV file")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs for encoder-decoder model training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for encoder-decoder model training")
    parser.add_argument('--save_endecoder', type=str, default='endecoder_model.pt', help="Path of encoder-decoder model weights")
    parser.add_argument('--load_endecoder', type=str, default='endecoder_model.pt', help="Path of encoder-decoder model weights to load for generation")


    # For generate
    parser.add_argument('--generate_data_csv', type=str, default='data_csv/data_e43V.csv',
                        help="Path to the input data CSV file")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of structures to generate")
    parser.add_argument("--generation_mode", type=str, default="full_guided", choices=["full_guided", "concat_only"], help="Generation strategy during inference.")

    # Condition parameters for generation (note: melting_point is log-scale in data)
    parser.add_argument('--cond_melting', type=float, default=np.log(1400.0), help="Desired melting point (log K)")
    parser.add_argument('--cond_density', type=float, default=8.0, help="Desired density (g/cc)")
    # parser.add_argument('--cond_form_energy', type=float, default=0.0, help="Desired formation energy (eV/atom)")
    parser.add_argument('--t', type=int, default=1000, help="...")
    # parser.add_argument('--guidance_scale', type=float, default=1.5, help="Classifier-free guidance scale for sampling (>1 for stronger conditioning)")
    args = parser.parse_args()

    mode = args.mode
    # Define columns in CSV for condition properties and extra features
    cond_cols = ["melting_point_log", "density", "formation_energy_per_atom"]
    extra_feature_cols = ["num_of_atoms", "energy_above_hull", "band_gap", "charge",
                           "electronic_energy", "total_enthalpy", "total_entropy",
                           "dielectric_constant", "refractive_index", "stoichiometry_sum", "volume_per_atom"]
    
    
    MAX_ATOMS = 20
    NUM_ATOM_TYPES = 44
    ALLOWED_ELEMENTS = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca",
                            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
                            "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", 
                            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", 
                            "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Hf",
                            "Ta", "W", "Re", "Os", "C", "N", "B", "Si"]
    """
    MAX_ATOMS = 10
    NUM_ATOM_TYPES = 17
    ALLOWED_ELEMENTS = ["Sc", "Ti", "V", "Cr", 
                            "Y", "Zr", "Nb", "Mo", "Hf",
                            "Ta", "W", "O", "C", "N", "B", "Si"]
    """

    if mode == 'pretrain':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and preprocess data
        df = pd.read_csv(args.pretrain_data_csv)
        df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))

        print(f"[INFO] Total structures before filtering: {len(df)}")

        df["is_valid"] = df["structure"].apply(is_valid_structure)
        df = df[df["is_valid"]].reset_index(drop=True)

        print(f"[INFO] Total valid structures after filtering: {len(df)}")

        # df.drop(columns=["is_valid"]).to_csv(f"cleaned_{args.pretrain_data_csv}", index=False)

        # Filter out samples that exceed MAX_ATOMS
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)
        print(f"[INFO] Filtered dataset: {len(df)} samples with ≤ {MAX_ATOMS} atoms")

        # === Filter structures to only contain ALLOWED_ELEMENTS ===
        allowed_Z = set(Element(sym).Z for sym in ALLOWED_ELEMENTS)
        def _all_sites_allowed(struct: Structure):
            try:
                return all(site.specie.Z in allowed_Z for site in struct)
            except Exception:
                return False
        before_allowed = len(df)
        df = df[df["structure"].apply(_all_sites_allowed)].reset_index(drop=True)
        print(f"[INFO] After ALLOWED_ELEMENTS filter ({len(ALLOWED_ELEMENTS)} elems): {len(df)} / {before_allowed} samples remain")

        # 1) Collect all occurring element Z values
        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums)) 
        print("Distinct elements (by atomic Z) in the dataset:", all_atomic_nums)
        print("Count =", len(all_atomic_nums))

        # 2) Build a dictionary mapping ‘Z → new class ID’
        # Keep 0 reserved for vacancies
        z2index = {}
        for i, Z in enumerate(all_atomic_nums, start=1):
            z2index[Z] = i

        print("z2index mapping:", z2index)
        # num_atom_types = len(all_atomic_nums) + 1  # +1 means vacancy = 0
        num_atom_types = NUM_ATOM_TYPES

        # num_atom_types = NUM_ATOM_TYPES
        print("num_atom_types =", num_atom_types)

        # Initialize dataset
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        print(f"Total samples: {len(df)}")
        print(f"Training set: {len(df_train)}, Validation set: {len(df_val)}")
        # dataset = StructureDataset(df, cond_cols=cond_cols, extra_feature_cols=extra_feature_cols)
        train_dataset = StructureDataset(
            df_train, 
            cond_cols=cond_cols, 
            extra_feature_cols=extra_feature_cols, 
            max_atoms=MAX_ATOMS, 
            z2index=z2index
        )
        # Extract all Wyckoff labels from the entire dataset
        all_labels = set()
        for idx in range(len(train_dataset)):
            wp_list = train_dataset[idx]["wyckoff_labels"]
            all_labels.update(wp_list)

        unique_labels = sorted(all_labels - {'0'})  # Delete '0'
        label2index = {'0': 0}  # Vacancy
        for i, label in enumerate(unique_labels, start=1):
            label2index[label] = i

        val_dataset = StructureDataset(
            df_val, 
            cond_cols=cond_cols, 
            extra_feature_cols=extra_feature_cols, 
            max_atoms=MAX_ATOMS, 
            z2index=z2index,
            cond_mean=train_dataset.cond_mean,
            cond_std=train_dataset.cond_std
        )
        
        # Use GPU for latent computation if available
        if torch.cuda.is_available():
            train_dataset.device = torch.device('cuda')
            val_dataset.device = torch.device('cuda')

        latent_dim = 64 + len(extra_feature_cols)  # 64 (graph embed dim) + extra features
        cond_dim = len(cond_cols)
        # Determine max_atoms in dataset and number of atom types (for decoder)
        # max_atoms = MAX_ATOMS

        # Initialize models
        decoder = StructureDecoder(
            latent_dim=latent_dim, 
            max_atoms=MAX_ATOMS, 
            num_atom_types=num_atom_types,
            z2index=z2index, 
            num_wp_labels=len(label2index),
            allowed_elements=ALLOWED_ELEMENTS
        )
        decoder.index2z = {v: k for k, v in z2index.items()}
        prop_model = PropertyPredictor(latent_dim=latent_dim, cond_dim=cond_dim)
        endecoder = EncoderDecoderModel(
            latent_dim=latent_dim, 
            max_atom_num=MAX_ATOMS, 
            extra_fea_dim=len(extra_feature_cols), 
            decoder=decoder, 
            prop_pred=prop_model, 
            num_atom_types=num_atom_types
        )

        # 3. Transfer the model to GPU/CPU
        endecoder.encoder.to(device)               
        decoder.to(device)              
        prop_model.to(device)
        endecoder.to(device)           

        
        # Train decoder and property predictor using training data
        # endecoder, loss_list, class_weights = train_encoder_decoder(
        #     endecoder, 
        #     train_dataset, 
        #     label2index=label2index,
        #     epochs=args.epochs, 
        #     batch_size=args.batch_size,
        #     learning_rate=1e-4,
        #     device=device,
        #     lattice_w=10.0, coord_w=5.0, type_w=10.0, prop_w=0.0,
        #     patience=20,
        #     num_atom_types=NUM_ATOM_TYPES,
        #     save_path=args.save_endecoder
        # )
        # plot_training_losses(loss_list, "Pretrain-Loss.png")
        
        print(f"[INFO] Loading pretrained weights from: {args.load_endecoder}")
        state_dict = torch.load(args.load_endecoder, map_location=device, weights_only=True)
        endecoder.load_state_dict(state_dict)

        
        class_weights = compute_class_weights(train_dataset, num_atom_types=NUM_ATOM_TYPES).to(device)

        val_metrics = evaluate_encoder_decoder(
            model=endecoder,
            dataset=val_dataset,
            label2index=label2index,
            batch_size=args.batch_size,
            device=device, 
            class_weights=class_weights,
            num_atom_types=NUM_ATOM_TYPES,
            lattice_w=10.0, coord_w=5.0, type_w=10.0, prop_w=0.0
        )

        print("Validation results:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
    

        print(f"Training completed.")
    
    elif mode == 'generate':        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(args.generate_data_csv)
        df = df.dropna()
        df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)
        print(f"[INFO] Filtered dataset: {len(df)} samples with ≤ {MAX_ATOMS} atoms")

        # === Filter structures to only contain ALLOWED_ELEMENTS ===
        allowed_Z = set(Element(sym).Z for sym in ALLOWED_ELEMENTS)

        def _all_sites_allowed(struct: Structure):
            try:
                return all(site.specie.Z in allowed_Z for site in struct)
            except Exception:
                return False

        before_allowed = len(df)
        df = df[df["structure"].apply(_all_sites_allowed)].reset_index(drop=True)
        print(f"[INFO] After ALLOWED_ELEMENTS filter ({len(ALLOWED_ELEMENTS)} elems): {len(df)} / {before_allowed} samples remain")

        df["is_valid"] = df["structure"].apply(is_valid_structure)
        df = df[df["is_valid"]].reset_index(drop=True)

        print(f"[INFO] Total valid structures after filtering: {len(df)}")
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

        if args.generation_mode == "full_guided":
            df_refer = df[
                (df["melting_point_log"] >= args.cond_melting) &
                (df["density"] <= args.cond_density)
            ].reset_index(drop=True)
            print(f"[INFO] Reference dataset filtered: {len(df_refer)} samples")
        else:
            df_refer = pd.DataFrame(columns=df.columns)
            print("[INFO] concat_only mode: skip reference-dataset-guided latent optimization.")

        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums))
        z2index = {Z: i + 1 for i, Z in enumerate(all_atomic_nums)}

        dataset = StructureDataset(
            df_train,
            cond_cols=cond_cols,
            extra_feature_cols=extra_feature_cols,
            max_atoms=MAX_ATOMS,
            z2index=z2index
        )

        refer_dataset = None
        if args.generation_mode == "full_guided":
            refer_dataset = StructureDataset(
                df_refer,
                cond_cols=cond_cols,
                extra_feature_cols=extra_feature_cols,
                max_atoms=MAX_ATOMS,
                z2index=z2index
            )

        all_labels = set()
        for idx in range(len(dataset)):
            wp_list = dataset[idx]["wyckoff_labels"]
            all_labels.update(wp_list)

        unique_labels = sorted(all_labels - {'0'})
        label2index = {'0': 0}
        for i, label in enumerate(unique_labels, start=1):
            label2index[label] = i

        if torch.cuda.is_available():
            dataset.device = torch.device('cuda')

        latent_dim = 64 + len(extra_feature_cols)
        cond_dim = len(cond_cols)
        num_atom_types = NUM_ATOM_TYPES

        # === Load Encoder-Decoder ===
        print("[INFO] Loading Encoder-Decoder model...")
        decoder = StructureDecoder(
            latent_dim=latent_dim,
            max_atoms=MAX_ATOMS,
            num_atom_types=num_atom_types,
            z2index=z2index,
            num_wp_labels=len(label2index),
            allowed_elements=ALLOWED_ELEMENTS
        )
        decoder.index2z = {v: k for k, v in z2index.items()}

        endecoder = EncoderDecoderModel(
            atom_fea_dim=64,
            edge_fea_dim=128,
            depth=3,
            max_atom_num=MAX_ATOMS,
            extra_fea_dim=len(extra_feature_cols),
            latent_dim=latent_dim,
            num_atom_types=num_atom_types,
            decoder=decoder,
            prop_pred=PropertyPredictor(latent_dim=latent_dim, cond_dim=cond_dim)
        )
        endecoder.load_state_dict(torch.load(args.load_endecoder, map_location=device, weights_only=True))
        endecoder.to(device).eval()

        # === Get training latents for novelty filter ===
        train_latents = []
        for i in range(len(dataset)):
            batch = dataset[i]
            with torch.no_grad():
                latent = endecoder.encoder(
                    batch["node_fea"].to(device),
                    batch["edge_index"].to(device),
                    batch["edge_fea"].to(device),
                    batch["crystal_atom_idx"].to(device),
                    batch["extra_fea"].to(device)
                ).squeeze(0)
            train_latents.append(latent.cpu().numpy())

        train_latents_np = np.array(train_latents)
        train_latents_tensor = torch.from_numpy(train_latents_np).float()

        # === Sample latent vectors ===
        latent_samples = torch.randn(args.num_samples, latent_dim, device=device)

        # === full_guided vs concat_only split ===
        if args.generation_mode == "full_guided":
            refer_lattices = []
            for i in range(len(refer_dataset)):
                refer_lattices.append(refer_dataset[i]["lattice_label"])
            refer_lattices = torch.stack(refer_lattices).to(device)  # [N, 6]

            N = min(len(df_refer), latent_samples.shape[0])
            assert refer_lattices.shape[0] >= N, "Not enough refer_lattices"
            refer_lattices = refer_lattices[torch.randperm(refer_lattices.shape[0])[:N]]

            opt_latents = latent_samples.clone()
            for i in range(N):
                z_i = latent_samples[i].clone().detach().requires_grad_(True)
                target_lattice_batch = refer_lattices[i]
                optimizer = torch.optim.Adam([z_i], lr=1e-2)

                for step in range(200):
                    _, lattice_norm, coords, types_logits = endecoder.decoder(z_i.unsqueeze(0), mode="generate")

                    lattice_loss = F.mse_loss(lattice_norm[0], target_lattice_batch)

                    types = types_logits[0].argmax(dim=-1)
                    mask = types > 0
                    c = coords[0][mask]
                    if c.size(0) > 1:
                        pair_dist = torch.cdist(c.unsqueeze(0), c.unsqueeze(0), p=2)[0]
                        min_dist = torch.topk(pair_dist.view(-1), 2, largest=False).values[1]
                        dist_loss = F.relu(0.5 - min_dist) ** 2
                    else:
                        dist_loss = torch.tensor(0.0, device=z_i.device)

                    loss = lattice_loss + 0.5 * dist_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                opt_latents[i] = z_i.detach()

            latent_samples = opt_latents
            print("[INFO] full_guided mode: latent optimization finished.")

        elif args.generation_mode == "concat_only":
            print("[INFO] concat_only mode: skip latent optimization and directly decode sampled latent.")

        # === Decode all sampled / optimized latents ===
        with torch.no_grad():
            _, lattice_norm, coords, types_logits = endecoder.decoder(latent_samples, mode="generate")

        A_MIN, A_MAX = 3.0, 18.0
        ALPHA_MIN, ALPHA_MAX = 45.0, 135.0

        def map_lattice_params(lattice_norm_np):
            a_raw, b_raw, c_raw, alpha_raw, beta_raw, gamma_raw = lattice_norm_np
            a = A_MIN + (A_MAX - A_MIN) * a_raw
            b = A_MIN + (A_MAX - A_MIN) * b_raw
            c = A_MIN + (A_MAX - A_MIN) * c_raw
            alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * alpha_raw
            beta  = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * beta_raw
            gamma = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * gamma_raw
            return a, b, c, alpha, beta, gamma

        def is_pathological_lattice(lattice):
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            vol = lattice.volume
            cond_number = np.linalg.cond(lattice.matrix)

            if min(a, b, c) < 1.0 or max(a, b, c) > 50.0:
                print("Unreasonable lattice lengths.")
                return True
            if any(angle < 10.0 or angle > 170.0 for angle in (alpha, beta, gamma)):
                print("Unreasonable lattice angles.")
                return True
            if vol < 5.0 or not math.isfinite(vol):
                print(f"Unreasonable lattice volume: {vol}")
                return True
            if cond_number > 1e6:
                print(f"High lattice condition number (ill-conditioned): {cond_number}")
                return True
            return False

        ###################################################################
        # Main Loop
        ###################################################################

        results = []
        unique_structures = []   # FIX: move outside loop so dedup really works
        matcher = StructureMatcher()

        print(f"[INFO] Start generating structures in {args.generation_mode} mode...")

        NOVELTY_THRESHOLD = 0.2
        for i in range(args.num_samples):
            print(f"--- SAMPLE {i + 1} ---")
            z = latent_samples[i].detach().cpu()
            dist_to_train = ((train_latents_tensor - z) ** 2).sum(dim=1).sqrt().min()

            if dist_to_train < NOVELTY_THRESHOLD:
                print(f"[Skip] Sample {i} too close to training set (dist = {dist_to_train:.4f})")
                continue

            lattice_params = map_lattice_params(lattice_norm[i].cpu().numpy())
            try:
                lattice = PmgLattice.from_parameters(*lattice_params)
                if is_pathological_lattice(lattice):
                    continue
            except Exception as e:
                print(f"[ERROR] Invalid lattice: {e}")
                continue

            coords_np = coords[i].cpu().numpy()
            types_np = types_logits[i].argmax(dim=-1).cpu().numpy()

            atom_indices = np.where(types_np != 0)[0].tolist()
            species, frac_coords = [], []

            for j in atom_indices:
                try:
                    type_idx = int(types_np[j])
                    Z = endecoder.decoder.index2z.get(type_idx, None)
                    if Z is None:
                        continue
                    el = Element.from_Z(int(Z))
                    if el.Z in allowed_Z:
                        species.append(el)
                        frac_coords.append(coords_np[j].tolist())
                except Exception as e:
                    print(f"[Warning] Skipping atom {j}: {e}")

            if len(set(species)) <= 1 or len(species) > 20:
                print(f"[Skip] Rejected structure due to atom count = {len(species)}")
                continue
            if len(set(species)) > 6:
                print(f"[Skip] Rejected structure due to too many atom types: {len(set(species))}, elements = {[e.symbol for e in set(species)]}")
                continue

            try:
                struct = Structure(lattice, species, frac_coords)
                dist = struct.distance_matrix
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.1:
                    continue

                formula = struct.composition.reduced_formula
                struct_json = json.dumps(struct.as_dict())

                def is_sun_structure(struct, unique_structures):
                    try:
                        analyzer = SpacegroupAnalyzer(struct, symprec=1e-2)
                        std_struct = analyzer.get_conventional_standard_structure()

                        if not std_struct.is_ordered:
                            return False

                        for s in unique_structures:
                            if matcher.fit(std_struct, s):
                                return False

                        unique_structures.append(std_struct)
                        return True
                    except Exception:
                        return False

                if is_sun_structure(struct, unique_structures):
                    sga = SpacegroupAnalyzer(struct, symprec=1e-2)
                    spacegroup = sga.get_space_group_symbol()
                    results.append({
                        "generation_mode": args.generation_mode,
                        "formula_pretty": formula,
                        "structure": struct_json,
                        "spacegroup": spacegroup
                    })

                # Save CIF file
                os.makedirs("generated_CIF", exist_ok=True)
                cif_filename = f"generated_CIF/generated_{args.generation_mode}_{i+1}_{formula}.cif"
                try:
                    CifWriter(struct).write_file(cif_filename)
                except Exception as e:
                    print(f"Warning: Failed to write CIF for sample {i+1}: {e}")

            except Exception as e:
                print(f"[ERROR] Failed to create Structure: {e}")

        # === Save results only once after loop ===
        if not results:
            print("No structures generated after filtering.")
        else:
            BEST_MODEL_PATHS = [
                "modelB-weights/best_modelB_mp_finetuned.pth",
                "modelB-weights/best_modelB_rho_finetuned.pth"
            ]
            TARGET_COLS = ["melting_point_log", "density"]

            df_new = pd.DataFrame(results)
            df_new["structure"] = df_new["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))

            df_pred = pd.DataFrame()
            df_pred["generation_mode"] = df_new["generation_mode"]
            df_pred["formula_pretty"] = df_new["formula_pretty"]
            df_pred["structure"] = df_new["structure"]
            df_pred["spacegroup"] = df_new["spacegroup"]

            # Load CGCNN model and predict
            def predict_values(df_struct, model_path):
                model = CGCNN(num_atom_types=100, use_extra_fea=False, extra_fea_dim=0)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.to(device)
                model.eval()

                all_atomic_nums = set()
                for struct in df_struct["structure"]:
                    for site in struct:
                        all_atomic_nums.add(site.specie.Z)
                all_atomic_nums = sorted(list(all_atomic_nums))
                z2index = {Z: i + 1 for i, Z in enumerate(all_atomic_nums)}

                def predict_single(structure):
                    node_fea, edge_index, edge_fea = build_cgcnn_graph(structure, z2index)
                    node_fea = torch.LongTensor(node_fea).to(device).unsqueeze(0)
                    edge_index = torch.LongTensor(edge_index).to(device)
                    edge_fea = torch.FloatTensor(edge_fea).to(device).unsqueeze(0)
                    crystal_atom_idx = torch.zeros(node_fea.shape[1], dtype=torch.long).to(device)

                    with torch.no_grad():
                        pred = model(node_fea[0], edge_index, edge_fea[0], crystal_atom_idx)
                    return pred.item()

                preds = []
                for i, row in df_struct.iterrows():
                    try:
                        preds.append(predict_single(row["structure"]))
                    except Exception as e:
                        print(f"[ERROR] Error at row {i} (formula_pretty: {row['formula_pretty']}): {e}")
                        preds.append(np.nan)
                return preds

            for BEST_MODEL_PATH, TARGET_COL in zip(BEST_MODEL_PATHS, TARGET_COLS):
                predictions = predict_values(df_new, BEST_MODEL_PATH)
                if TARGET_COL == "melting_point_log":
                    predictions = np.exp(predictions)
                    df_pred["pred_melting_point"] = predictions
                else:
                    df_pred["pred_density"] = predictions

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("generated_structures", exist_ok=True)
            output_csv = f"generated_structures/generated_structures_{args.generation_mode}_{timestamp}.csv"
            df_pred["structure"] = df_pred["structure"].apply(lambda x: json.dumps(x.as_dict()))
            df_pred.to_csv(output_csv, index=False)

            print(f"[FINISHED] Generated {len(results)} structures saved to {output_csv} (CIF files saved for each structure).")
            print("[FINISHED] Generation completed.")
            print("--END--")