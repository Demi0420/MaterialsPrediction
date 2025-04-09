import json
import os
import math
import argparse
import pandas as pd
import numpy as np
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pymatgen.core import Structure, Lattice, Element, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from sklearn.model_selection import train_test_split
import csv
import tqdm
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
                 use_extra_fea=True, extra_fea_dim=10):
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
            "coords_label": coords_pad_ts,  # [max_atoms,3]
            "types_label": types_pad_ts     # [max_atoms]
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
        latent_dim=64+11,  # 64 graph embed + 10 extra = 74
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
        # 这里假设 batch_size=1 per structure if you don't do a bigger multi-structure batch
        latent = self.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)  # shape [1, latent_dim]
        # decode
        lattice, lattice_norm, coords, types_logits = self.decoder(latent)
        prop_out = None
        if self.prop_pred is not None:
            prop_out = self.prop_pred(latent)
        return lattice, lattice_norm, coords, types_logits, prop_out
    


class DiffusionModel(nn.Module):
    """Conditional diffusion model that predicts noise given latent x_t, time t, and conditions."""
    def __init__(
            self, 
            latent_dim: int, 
            time_embed_dim: int = 32,
            hidden_dim: int = 256
        ):

        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim

        # Linear layers for time embedding, condition embedding, and latent input
        self.fc_time = nn.Linear(time_embed_dim, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)

        # Two hidden layers after combining embeddings
        self.fc_hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer to predict noise in latent space
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(
            self, 
            x_t: torch.Tensor, 
            t: torch.Tensor
        ):
        """
        Args:
            x_t: [batch, latent_dim] noised latent at time t.
            t: [batch] diffusion timestep indices.
            cond: [batch, cond_dim] condition vector (normalized target properties or zeros if uncond).
        Returns:
            noise_pred: [batch, latent_dim] predicted noise for x_t.
        """
        # batch_size = x_t.size(0)
        # Sinusoidal time embedding (positional encoding for t)
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)  # [batch, time_embed_dim]
        t_embed = F.relu(self.fc_time(t_embed))                      # [batch, hidden_dim]

        # Latent input projection
        x_embed = F.relu(self.fc_latent(x_t))                        # [batch, hidden_dim]
        # Combine embeddings (elementwise addition)
        
        h = x_embed + t_embed
        # Further processing through hidden layers
        h = F.relu(self.fc_hidden1(h))
        h = F.relu(self.fc_hidden2(h))

        # Output noise prediction
        noise_pred = self.fc_out(h)
        return noise_pred
    
    def reverse_sample(
        self,
        batch_size: int,
        T: int = 1000,
        device=torch.device("cpu")
    ):
        """
        Unconditional reverse diffusion sampling: generate latent vector z_0 from noise.

        Args:
            batch_size: number of samples to generate.
            T: number of diffusion steps.
            device: device to run on.

        Returns:
            z_0: [batch_size, latent_dim]
        """
        self.eval()
        latent_dim = self.latent_dim

        # Prepare diffusion schedule
        betas, alphas, alpha_cum = prepare_diffusion_schedule(T)
        betas, alphas, alpha_cum = betas.to(device), alphas.to(device), alpha_cum.to(device)

        # Start from pure Gaussian noise
        x_t = torch.randn((batch_size, latent_dim), device=device)
        eps = 1e-5  # numerical stability

        for t in reversed(range(T)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)
            noise_pred = self.forward(x_t, t_batch)

            # Get diffusion parameters
            beta_t = betas[t]                          # scalar
            alpha_t = alphas[t]                        # scalar
            alpha_bar_t = alpha_cum[t]                 # scalar
            alpha_bar_prev = alpha_cum[t - 1] if t > 1 else torch.tensor(1.0, device=device)

            # Reshape for broadcasting: [B, 1]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t + eps).unsqueeze(0).repeat(batch_size, 1)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t + eps).unsqueeze(0).repeat(batch_size, 1)

            # Predict x0 from x_t and predicted noise
            x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -10, 10)  # Optional clamp for stability

            if torch.isnan(x0_pred).any():
                print(f"[reverse_sample] NaN detected at timestep {t} in x0_pred")
                x0_pred = torch.nan_to_num(x0_pred, nan=0.0)

            # Compute mean of posterior q(x_{t-1} | x_t, x0)
            coef1 = (alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t + eps)).unsqueeze(0).repeat(batch_size, 1)
            coef2 = ((1 - alpha_bar_prev) * alpha_t.sqrt() / (1 - alpha_bar_t + eps)).unsqueeze(0).repeat(batch_size, 1)
            mean = coef1 * x0_pred + coef2 * x_t

            if t > 1:
                sigma = torch.sqrt((1 - alpha_bar_prev + eps) / (1 - alpha_bar_t + eps) * beta_t)
                noise = torch.randn_like(x_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean  # final step

            if torch.isnan(x_t).any():
                print(f"[reverse_sample] NaN detected at timestep {t} in x_t")
                x_t = torch.nan_to_num(x_t, nan=0.0)

        return x_t  # [B, latent_dim]



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
        "types_label": torch.stack(types_labels)
    }

def compute_class_weights(dataset, num_atom_types):
    # 统计所有样本的 type 标签
    all_types = []
    for sample in dataset:
        all_types.append(sample["types_label"].numpy())  # shape: [max_atoms]
    all_types = np.concatenate(all_types)  # shape: [N_total_atoms]
    
    # 统计每种 type 的出现次数
    type_counts = np.bincount(all_types, minlength=num_atom_types)

    # 计算权重 ∝ 1 / count（防止除0加一个小常数）
    class_weights = 1.0 / (type_counts + 1e-6)

    # 归一化（不是必须，但有时稳定）
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

def train_encoder_decoder(
    model: EncoderDecoderModel,
    dataset: StructureDataset,
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
            types_label = batch_data["types_label"].to(device)         # [max_atoms]

            # forward
            lattice, lattice_norm, coords, types_logits = model(
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
                    dist = (diff**2).sum(-1) + 1e-8         # 加稳定项，防止 sqrt(0)
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
                # type_loss = F.cross_entropy(
                #     pred_types_2d[valid_mask], 
                #     types_label.view(-1)[valid_mask], 
                #     weight=class_weights
                # )

                # === (1) 强制 class_weights[0] 极小，避免模型学到 type=0 ===
                class_weights[0] = 0.05  # padding 类型不重要，权重极小

                type_loss = focal_loss(
                    logits=pred_types_2d[valid_mask],
                    targets=types_label.view(-1)[valid_mask],
                    alpha=class_weights.to(device), 
                    gamma=1.5,                   # 可调整，越大越聚焦
                    # ignore_index=0               # 忽略 padding 类型
                )
                
                # === (3) 在 logits 中抑制 type=0，防止推理时预测为0 ===
                # types_logits[..., 0] = -1e9  # 这一步在训练和验证时都可以加

                type_count_loss = nonzero_type_penalty(types_logits, target_min=4)
                true_atom_counts = (types_label > 0).sum(dim=1).float()  # [B]
                pred_types = types_logits.argmax(dim=-1)
                pred_atom_counts = (pred_types != 0).sum(dim=1).float()
                count_loss = F.mse_loss(pred_atom_counts, true_atom_counts)

                
                # 3.5) type prediction distribution analysis (debugging collapse)
                with torch.no_grad():
                    pred_types_2d = torch.clamp(pred_types_2d, -30, 30)  # 限制 logits 范围，防止数值爆炸
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

            total_loss = (
                lattice_w * lattice_loss + 
                diversity_penalty + 
                coord_w * coord_loss + 
                0.2 * coord_diversity_loss +
                2.0 * dist_loss + 
                type_w * type_loss +
                -0.1 * type_entropy_bonus +
                0.5 * type_count_loss +
                1.0 * count_loss
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
            num_batches += 1

        avg_total_loss = epoch_total_loss / num_batches

        loss_history.append((
            avg_total_loss,
            epoch_lattice_loss / num_batches,
            epoch_coord_loss / num_batches,
            epoch_type_loss / num_batches,
            # epoch_prop_loss / num_batches
        ))

        print(f"Epoch {epoch}/{epochs}, total_loss={avg_total_loss:.4f}")
        print(f"Epoch {epoch}/{epochs}, Lattice Loss: {epoch_lattice_loss / num_batches:.4f} | "
              f"Coords Loss: {epoch_coord_loss / num_batches:.4f} | Type Loss: {epoch_type_loss / num_batches:.4f} | ")
              # f"Prop Loss: {epoch_prop_loss / num_batches:.4f}")
        

        if epoch % 1 == 0:
            # === 汇总 epoch 的预测类型统计 + 检查原子坐标 ===
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

                    _, _, coords_pred, types_logits_pred, _ = model(
                        node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
                    )

                    # === 收集类型分布统计（全 batch）===
                    pred_types_2d = types_logits_pred.view(-1, types_logits_pred.size(-1))
                    pred_type = pred_types_2d.argmax(dim=-1).cpu().numpy()
                    true_type = types_label.view(-1).cpu().numpy()
                    valid_mask = true_type != 0

                    epoch_pred_types.append(pred_type[valid_mask])
                    epoch_true_types.append(true_type[valid_mask])

                    # === 仅第一个 batch 可视化坐标等（防止太慢） ===
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

                            print(f"[Epoch {epoch}] Sample {b}: Predicted atom count = {valid_mask_pred.sum()}")
                            print(pred_t)
                            print(f"[Epoch {epoch}] Sample {b}: Ground truth atom count = {valid_mask_true.sum()}")
                            print(true_t)

                            print(f"[Epoch {epoch}] Sample {b}: Unique predicted types = {np.unique(pred_t[valid_mask_pred])}")
                            print(f"[Epoch {epoch}] Sample {b}: Unique ground truth types = {np.unique(true_t[valid_mask_true])}")
                            
                            if coords_valid_pred.shape[0] > 1:
                                dists_pred = np.linalg.norm(
                                    coords_valid_pred[:, None, :] - coords_valid_pred[None, :, :], axis=-1)
                                np.fill_diagonal(dists_pred, np.inf)
                                min_d_pred = np.min(dists_pred)
                                print(f"[Epoch {epoch}] Sample {b}: Predicted min_dist = {min_d_pred:.3f}")
                                print(f"[Epoch {epoch}] Sample {b} pred coords (first 3 atoms):\n", coords_valid_pred[:3])

                            if coords_valid_true.shape[0] > 1:
                                dists_true = np.linalg.norm(
                                    coords_valid_true[:, None, :] - coords_valid_true[None, :, :], axis=-1)
                                np.fill_diagonal(dists_true, np.inf)
                                min_d_true = np.min(dists_true)
                                print(f"[Epoch {epoch}] Sample {b}: Ground truth min_dist = {min_d_true:.3f}")
                                print(f"[Epoch {epoch}] Sample {b} true coords (first 3 atoms):\n", coords_valid_true[:3])

            atom_count_pred = [np.sum(p != 0) for p in pred_types]
            atom_count_true = [np.sum(t != 0) for t in true_types]
            errors = np.array(atom_count_pred) - np.array(atom_count_true)
            print(f"[Epoch {epoch}] Avg atom count error: {errors.mean():.2f}")


            # === 汇总类型分布可视化 ===
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
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.hist(errors, bins=np.arange(-MAX_ATOMS, MAX_ATOMS+2, 1), color='orange', edgecolor='black')
            plt.title(f"Epoch {epoch}: Atom Count Error Histogram (Predicted - Ground Truth)")
            plt.xlabel("Atom Count Error")
            plt.ylabel("Number of Samples")
            plt.tight_layout()
            plt.savefig(f"epoch_vis/epoch_{epoch:03d}_atom_count_error_hist.png", dpi=300)
            plt.close()

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

def sinusoidal_time_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Produce sinusoidal time-step embeddings (similar to positional embeddings).
    """
    # Compute sinusoidal position embeddings as in Transformer or DDPM
    device = timesteps.device
    half_dim = embedding_dim // 2
    # Create a range [0, half_dim) and compute inverse frequencies
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    # Embed timesteps to [batch, half_dim] using sinus and cos
    angles = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embedding_dim % 2 == 1:  # if odd, pad one zero dimension
        emb = F.pad(emb, (0, 1))
    return emb

# Probability for dropping condition (for classifier-free guidance training)
GUIDANCE_UNCOND_PROB = 0.1

def prepare_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)  # shape [T]
    return betas, alphas, alpha_cum


def sample_diffusion(
        model: DiffusionModel, 
        cond: torch.Tensor, 
        num_samples: int = 1, 
        T: int = 1000, 
        guidance_scale: float = 1.0, 
        device=torch.device('cpu')
    ):
    """
    Sample new latent vectors from the diffusion model given desired conditions.
    Uses classifier-free guidance during sampling if guidance_scale > 1.
    Args:
        model: Trained DiffusionModel.
        cond: Tensor shape [cond_dim] or [num_samples, cond_dim] of desired conditions (normalized).
        num_samples: Number of samples to generate.
        T: Total diffusion steps (should match training).
        guidance_scale: Scale for classifier-free guidance (1.0 = no guidance, >1 to strengthen conditioning).
        device: Torch device.
    Returns:
        Tensor of shape [num_samples, latent_dim] of generated latent vectors.
    """
    model.to(device)
    model.eval()
    # Ensure cond is shape [num_samples, cond_dim]
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    cond = cond.to(device)
    if cond.size(0) < num_samples:
        cond = cond.repeat(num_samples, 1)
    _, _, alpha_cum = prepare_diffusion_schedule(T)
    alpha_cum = alpha_cum.to(device)
    latent_dim = model.latent_dim
    samples = []
    for i in range(num_samples):
        x_t = torch.randn(1, latent_dim, device=device)  # start from standard normal latent
        for t in range(T-1, -1, -1):
            t_tensor = torch.tensor([t], device=device)
            # Classifier-free guidance: obtain noise prediction for cond and for uncond (zero cond)
            if guidance_scale > 1.0:
                cond_vec = cond[i:i+1]
                uncond_vec = torch.zeros_like(cond_vec)
                noise_pred_cond = model(x_t, t_tensor, cond_vec)
                noise_pred_uncond = model(x_t, t_tensor, uncond_vec)
                # Combine predictions with guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No guidance, use conditional prediction directly
                noise_pred = model(x_t, t_tensor, cond[i:i+1])
            # Diffusion posterior mean for x_{t-1}
            alpha_cum_t = alpha_cum[t]
            alpha_cum_prev = alpha_cum[t-1] if t-1 >= 0 else torch.tensor(1.0, device=device)
            x0_pred = (1/torch.sqrt(alpha_cum_t)) * (x_t - torch.sqrt(1 - alpha_cum_t) * noise_pred)
            mean = torch.sqrt(alpha_cum_prev) * x0_pred + torch.sqrt(1 - alpha_cum_prev) * noise_pred
            if t > 1:
                # Sample noise for transition to t-1
                beta_t = 1 - (alpha_cum[t] / alpha_cum_prev)
                z = torch.randn_like(x_t, device=device)
                x_t = mean + torch.sqrt(beta_t) * z
            else:
                # t == 1, no noise added
                x_t = mean
        samples.append(x_t.squeeze(0))
    samples = torch.stack(samples, dim=0)  # [num_samples, latent_dim]
    return samples


class StructureDecoder(nn.Module):
    """
    Learnable decoder that generates crystal structure (lattice parameters, atomic coordinates, types)
    from a latent vector.
    Outputs:
      - Lattice parameters (a, b, c, alpha, beta, gamma)
      - Atomic fractional coordinates for max_atoms sites
      - Atomic type probabilities for max_atoms sites

    在原有基础上做了如下改动：
      - 不再直接输出 (a,b,c,alpha,beta,gamma)，而是输出 "raw_lattice" 6维
      - 用 sigmoid 将其映射到 [0,1], 再缩放到合理区间 [self.a_min, self.a_max], [self.alpha_min, self.alpha_max], ...
      - 避免出现无效晶格角度和负值
    """
    def __init__(
            self, 
            latent_dim: int, 
            max_atoms: int, 
            num_atom_types: int, 
            a_min=3.0, a_max=18.0, alpha_min=45.0, alpha_max=135.0,
            # hidden_dim: int = 256
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types

        # 这里示例性地设定了 a,b,c 的最小最大值，alpha,beta,gamma 的最小最大值
        # 需要根据自己大数据集统计得到更真实的范围
        self.a_min = a_min
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max


        # 随意写了隐藏维度, 具体可根据需求修改
        hidden_dim = 256

        # 用全连接层把 latent -> lattice + coords + types
        self.lattice_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # raw_lattice, 6个分量
        )

        # 预测坐标 [max_atoms, 3]
        self.coord_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms * 3)
        )

        # 预测 atom types [max_atoms, num_atom_types]
        self.type_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_atoms * num_atom_types)
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [batch, latent_dim] latent vectors.
        Returns:
            lattice_out: [batch, 6] predicted (a, b, c, alpha_frac, beta_frac, gamma_frac) 
                         (angles are fractional of 180 degrees).
            coords_out: [batch, max_atoms, 3] predicted fractional coordinates.
            types_out: [batch, max_atoms, num_atom_types] logits for atomic types (0 = empty).
        """
        batch_size = z.size(0)

        # 1) 预测晶格参数 (raw)
        raw_lattice = self.lattice_fc(z)  # [batch_size, 6]
        # 拆分
        raw_a, raw_b, raw_c, raw_alpha, raw_beta, raw_gamma = torch.chunk(raw_lattice, chunks=6, dim=1)

        # === 使用 tanh 映射到 [-1,1] 后再映射到目标范围 ===
        a_tanh     = torch.tanh(raw_a)
        b_tanh     = torch.tanh(raw_b)
        c_tanh     = torch.tanh(raw_c)
        alpha_tanh = torch.tanh(raw_alpha)
        beta_tanh  = torch.tanh(raw_beta)
        gamma_tanh = torch.tanh(raw_gamma)

        # tanh ∈ [-1, 1] → 映射到 [min, max]
        def map_range(val, min_val, max_val):
            return 0.5 * (val + 1.0) * (max_val - min_val) + min_val

        a = map_range(a_tanh, self.a_min, self.a_max)
        b = map_range(b_tanh, self.a_min, self.a_max)
        c = map_range(c_tanh, self.a_min, self.a_max)
        alpha = map_range(alpha_tanh, self.alpha_min, self.alpha_max)
        beta  = map_range(beta_tanh, self.alpha_min, self.alpha_max)
        gamma = map_range(gamma_tanh, self.alpha_min, self.alpha_max)

        # 再映射回 [0,1] 版本用于训练用 norm 表征（可选）
        a_01     = (a - self.a_min) / (self.a_max - self.a_min)
        b_01     = (b - self.a_min) / (self.a_max - self.a_min)
        c_01     = (c - self.a_min) / (self.a_max - self.a_min)
        alpha_01 = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
        beta_01  = (beta - self.alpha_min) / (self.alpha_max - self.alpha_min)
        gamma_01 = (gamma - self.alpha_min) / (self.alpha_max - self.alpha_min)

        a = self.a_min + (self.a_max - self.a_min) * a_01
        b = self.a_min + (self.a_max - self.a_min) * b_01
        c = self.a_min + (self.a_max - self.a_min) * c_01

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_01
        beta  = self.alpha_min + (self.alpha_max - self.alpha_min) * beta_01
        gamma = self.alpha_min + (self.alpha_max - self.alpha_min) * gamma_01

        # 拼回 lattice 的物理值
        lattice = torch.cat([a, b, c, alpha, beta, gamma], dim=1)  # shape [B,6]

        # 同时 lattice_norm 也是 [0,1] (跟 train_decoder里的 'lattice_batch' 对齐)
        lattice_norm = torch.cat([a_01, b_01, c_01, alpha_01, beta_01, gamma_01], dim=1)  # [B,6]

        # ============= 2) 坐标coords (fractional) =============
        coords_raw = self.coord_fc(z)  # [B, max_atoms*3]
        coords_raw = coords_raw.view(batch_size, self.max_atoms, 3)
        # 使用 tanh 替代 sigmoid，避免 collapse，映射到 [0, 1]
        coords_tanh = torch.tanh(coords_raw)  # [B, max_atoms, 3]
        coords = 0.5 * (coords_tanh + 1.0) # 映射到 [0, 1]

        # 训练时加入轻微扰动，增加多样性（可选）
        if self.training:
            noise = torch.randn_like(coords) * 0.01  # 标准差可调
            coords = coords + noise
            coords = torch.clamp(coords, 0.0, 1.0)  # 保证仍在合法范围内

        # ============= 3) 原子类型 logits =============
        types_raw = self.type_fc(z).view(batch_size, self.max_atoms, self.num_atom_types)
        # types_raw 直接作为 logits
        types_logits = types_raw

        return lattice, lattice_norm, coords, types_logits

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
            _, lattice_norm, coords, types_logits, prop_out = model(
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
                    dist = (diff**2).sum(-1) + 1e-8         # 加稳定项，防止 sqrt(0)
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
                    gamma=1.5,                   # 可调整，越大越聚焦
                    # ignore_index=0               # 忽略 padding 类型
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

                # 记录预测/真实类型
                pred_type = pred_type_probs.argmax(dim=-1).cpu().numpy()
                true_type = types_label.view(-1)[valid_mask].cpu().numpy()
                all_pred_types.extend(pred_type)
                all_true_types.extend(true_type)

            # 4) property loss
            # prop_loss = 0.0
            # if prop_out is not None:
            #     prop_loss = F.mse_loss(prop_out, cond, reduction='mean')
            #     prop_losses.append(prop_loss.item())


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
                1.0 * count_loss
                # prop_w * prop_loss
            )
            total_losses.append(total_loss.item())
        


        # === 可视化前 30 个样本的预测结果 ===
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

            lattice, lattice_norm, coords_pred, types_logits_pred, _ = model(
                node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea
            )

            lattices_pred = lattice_norm.cpu().numpy()         # shape [6], normalized lattice
            lattices_true = lattice_label.cpu().numpy()        # shape [6], normalized lattice

            
            pred_types = types_logits_pred.argmax(dim=-1).cpu().numpy()  # [B, max_atoms]
            coords_np = coords_pred.cpu().numpy()                         # [B, max_atoms, 3]
            coords_true = coords_label.cpu().numpy()                      # [B, max_atoms, 3]
            true_types = types_label.cpu().numpy()                        # [B, max_atoms]

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

                sample_count += 1
            if sample_count >= 30:
                break

    # === 打印预测类型分布 ===
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
    }


def plot_training_losses(loss_history, figure_path):
    """
    Expects loss_history to be a list of tuples:
    (total_loss, lattice_loss, coord_loss, type_loss, prop_loss)
    Only plot the first 4 loss terms.
    """
    total_loss_list = [l[0] for l in loss_history]
    lattice_loss_list = [l[1] for l in loss_history]
    coord_loss_list = [l[2] for l in loss_history]
    type_loss_list = [l[3] for l in loss_history]
    # prop_loss_list = [l[4] for l in loss_history]

    epochs = list(range(1, len(loss_history)+1))

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].plot(epochs, total_loss_list, label='Total Loss', color='blue')
    axs[0].set_title("Total Loss")

    axs[1].plot(epochs, lattice_loss_list, label='Lattice Loss', color='green')
    axs[1].set_title("Lattice Loss")

    axs[2].plot(epochs, coord_loss_list, label='Coord Loss', color='orange')
    axs[2].set_title("Coord Loss")

    axs[3].plot(epochs, type_loss_list, label='Type Loss', color='red')
    axs[3].set_title("Type Loss")

    # axs[4].plot(epochs, prop_loss_list, label='Prop Loss', color='black')
    # axs[4].set_title("Prop Loss")

    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    # plt.show()


def plot_diffusion_losses(loss_history, figure_path):
    epochs = list(range(1, len(loss_history) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_history, label='Diffusion Loss', color='black')
    
    plt.title("Training Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
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
        return new_atom_fea  # 残差 + softplus
    

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

        # MLP 输出
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


# ===== 平滑控制 sampling 概率与 loss 权重 =====
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

    # 1. Lattice 可视化
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

    # 2. Type 分布对比
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
    """
    打印每个 batch 样本的 lattice、type、coords 预测与真实对比。

    Args:
        epoch: 当前 epoch
        coords_pred: Tensor [B, max_atoms, 3]
        coords_label: Tensor [B, max_atoms, 3]
        lattice_pred: Tensor [B, 6]
        lattice_label: Tensor [B, 6]
        types_logits: Tensor [B, max_atoms, num_types]
        types_label: Tensor [B, max_atoms]
        visualize_fn: 可选函数 visualize(lattice_pr, lattice_gt, pred_types, true_types, epoch)
    """
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

def train_diffusion_model_unconditional(
    model: DiffusionModel,
    encoder,
    decoder,
    dataset: StructureDataset,
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: int = 1e-4,
    T: int = 1000,
    device=torch.device('cpu'),
    save_path: str = "final_diffusion_model_uncond.pt",
    lambda_lattice: float = 10.0,
    warmup_epochs: int = 10
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.to(device)
    encoder.to(device)
    decoder.to(device)

    model.train()
    encoder.eval()
    decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    _, _, alpha_cum = prepare_diffusion_schedule(T)
    alpha_cum = alpha_cum.to(device)

    latent_mean, latent_std = compute_latent_global_stats(encoder, dataset, device, batch_size, collate_fn)

    epoch_losses = []

    for epoch in range(1, epochs + 1):
        # sampling_prob = warmup_cosine(epoch, warmup_epochs, epochs, max_val=0.8, min_val=0.0)
        # lambda_lattice_dyn = warmup_cosine(epoch, warmup_epochs, epochs, max_val=lambda_lattice, min_val=0.3)
        # lambda_recon_dyn = warmup_cosine(epoch, warmup_epochs, epochs, min_val=0.3, max_val=1.0)
        sampling_prob = 0.0  # 强制只用 latent_gt
        lambda_lattice_dyn = lambda_lattice
        lambda_recon_dyn = lambda_recon

        batch_losses, recon_losses, lattice_losses, type_entropys = [], [], [], []

        for batch_data in dataloader:
            node_fea = batch_data["node_fea"].to(device)
            edge_index = batch_data["edge_index"].to(device)
            edge_fea = batch_data["edge_fea"].to(device)
            crystal_atom_idx = batch_data["crystal_atom_idx"].to(device)
            lattice_label = batch_data["lattice_label"].to(device)

            with torch.no_grad():
                latent_gt = encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, batch_data["extra_fea"].to(device))
                latent_gt = (latent_gt - latent_mean) / latent_std

            B = latent_gt.size(0)
            t_batch = torch.randint(1, T, (B,), device=device)
            noise = torch.randn_like(latent_gt)

            alpha_cum_t = alpha_cum[t_batch].unsqueeze(1)
            x_t = torch.sqrt(alpha_cum_t) * latent_gt + \
                  torch.sqrt(1.0 - alpha_cum_t) * noise

            noise_pred = model(x_t, t_batch)
            loss_recon = F.mse_loss(noise_pred, noise)

            do_sampling = np.random.rand() < sampling_prob
            if do_sampling:
                with torch.no_grad():
                    z_gen = model.reverse_sample(batch_size=B, T=T, device=device)
                    if torch.isnan(z_gen).any():
                        print("[WARN] z_gen contains NaN! Skipping batch.")
                        continue
                    _, lattice_pred, coords_pred, types_logits = decoder(z_gen)
                    pred_type_probs = F.softmax(types_logits, dim=-1)
                    entropy = - (pred_type_probs * torch.log(pred_type_probs + 1e-8)).sum(dim=-1)
                    valid_mask = (types_logits.max(dim=-1).values > 0).float()
                    type_entropy = (entropy * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            else:
                with torch.no_grad():
                    z_used = latent_gt
                    _, lattice_pred, coords_pred, types_logits = decoder(z_used)
                type_entropy = torch.tensor(0.0, device=device)

            if torch.isnan(lattice_pred).any():
                print("[ERROR] NaN in lattice prediction, skipping batch.")
                continue

            lattice_loss = F.mse_loss(lattice_pred, lattice_label)
            lattice_loss = torch.clamp(lattice_loss, max=1.0)

            total_loss = (
                lambda_recon_dyn * loss_recon +
                lambda_lattice_dyn * lattice_loss +
                (- 0.05 * type_entropy if do_sampling else 0.0)
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_losses.append(total_loss.item())
            recon_losses.append(loss_recon.item())
            lattice_losses.append(lattice_loss.item())
            type_entropys.append(type_entropy.item())

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch}/{epochs} | Total: {epoch_loss:.4f} | Recon: {np.mean(recon_losses):.4f} | "
              f"Lattice: {np.mean(lattice_losses):.4f} (λ={lambda_lattice_dyn:.2f}) | "
              f"Entropy: {np.mean(type_entropys):.4f} | Sampling Prob: {sampling_prob:.2f}")
        
        if epoch % 1 == 0:
            evaluate_and_log_diffusion_outputs(
                epoch,
                coords_pred,
                batch_data["coords_label"],
                lattice_pred,
                batch_data["lattice_label"],
                types_logits,
                batch_data["types_label"]
                # visualize_fn=visualize_lattice_and_type 
            )

    torch.save(model.state_dict(), save_path)
    print(f"[✔] Final model saved to {save_path}")

    return model, epoch_losses


def evaluate_diffusion_model(
    encoder,
    decoder,
    diffusion_model,
    prop_model,
    dataset,
    num_samples=100,
    T=1000,
    guidance_scale=1.0,
    device=torch.device("cpu")
):
    encoder.eval()
    decoder.eval()
    diffusion_model.eval()
    prop_model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    results = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break

            cond = batch["cond"].to(device)  # shape: [1, cond_dim]
            result_entry = {
                "structure_id": count,
                "cond_input": cond.squeeze(0).cpu().tolist(),
            }

            try:
                # Generate latent vector
                z_sample = sample_diffusion(
                    diffusion_model,
                    cond[0],  # [cond_dim]
                    num_samples=1,
                    T=T,
                    guidance_scale=guidance_scale,
                    device=device
                )[0]  # [latent_dim]

                # Decode structure
                lattice_pred, lattice_norm, coords_pred, types_logits = decoder(z_sample.unsqueeze(0))
                type_pred = types_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                atom_indices = [i for i, t in enumerate(type_pred) if t != 0]

                if len(atom_indices) == 0:
                    print(f"Sample {i+1} filtered out (no allowed elements)")
                    result_entry["status"] = "fail"
                    result_entry["reason"] = "no_atoms_predicted"
                    results.append(result_entry)
                    continue

                # Build lattice
                lattice = Lattice.from_parameters(*lattice_pred.squeeze(0).cpu().tolist())

                # Build structure
                species = []
                coords = []
                for i in atom_indices:
                    Z = decoder.index2z[int(type_pred[i])]
                    species.append(Element.from_Z(Z))
                    coords.append(coords_pred[0, i].cpu().numpy())

                structure = Structure(lattice, species, coords, coords_are_cartesian=False)

                # Check for minimum number of atoms
                if len(species) < 4:
                    result_entry["status"] = "fail"
                    result_entry["reason"] = f"too_few_atoms={len(species)}"
                    results.append(result_entry)
                    continue

                if len(species) > 20:
                    result_entry["status"] = "fail"
                    result_entry["reason"] = f"too_many_atoms={len(species)}"
                    results.append(result_entry)
                    continue

                # Distance check
                dist_mat = structure.distance_matrix
                np.fill_diagonal(dist_mat, np.inf)
                min_dist = dist_mat.min()

                if min_dist < 1.0:
                    result_entry["status"] = "fail"
                    result_entry["reason"] = f"atom_overlap_min_dist={min_dist:.4f}"
                    results.append(result_entry)
                    continue

                # Property prediction
                pred_prop = prop_model(z_sample.unsqueeze(0))  # [1, cond_dim]
                prop_mse = F.mse_loss(pred_prop, cond).item()

                # Save success entry
                result_entry.update({
                    "status": "success",
                    "prop_mse": prop_mse,
                    "min_interatomic_distance": float(min_dist),
                    "num_atoms": len(species),
                    "elements": [str(el) for el in species],
                    "lattice_matrix": lattice.matrix.tolist(),
                    "frac_coords": np.array(coords).tolist()
                })

                results.append(result_entry)
                count += 1

            except Exception as e:
                result_entry["status"] = "fail"
                result_entry["reason"] = str(e)
                results.append(result_entry)
                continue

    # === Save results ===
    with open("diffusion_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save summary to CSV (only success entries)
    try:
        success_entries = [r for r in results if r["status"] == "success"]
        df_results = pd.DataFrame({
            "structure_id": [r["structure_id"] for r in success_entries],
            "prop_mse": [r["prop_mse"] for r in success_entries],
            "min_interatomic_distance": [r["min_interatomic_distance"] for r in success_entries],
            "num_atoms": [r["num_atoms"] for r in success_entries],
        })
        df_results.to_csv("evaluation_results.csv", index=False)
        print("[INFO] Saved evaluation results.")
    except Exception as e:
        print(f"[WARNING] Failed to save evaluation summary CSV: {e}")

    # === Print summary ===
    success = len([r for r in results if r["status"] == "success"])
    fail = len(results) - success
    print(f"\n[Evaluation Results]")
    print(f"Success: {success}")
    print(f"Fail: {fail}")
    if success > 0:
        print(f"Average property MSE: {np.mean([r['prop_mse'] for r in results if r['status'] == 'success']):.4f}")
        print(f"Average min interatomic distance: {np.mean([r['min_interatomic_distance'] for r in results if r['status'] == 'success']):.4f}")
        print(f"Average number of atoms: {np.mean([r['num_atoms'] for r in results if r['status'] == 'success']):.2f}")

    return results

def is_valid_structure(struct: Structure) -> bool:
    try:
        # 检查晶格参数是否为有限值
        lengths = struct.lattice.abc  # (a, b, c)
        angles = struct.lattice.angles  # (alpha, beta, gamma)
        if not all(np.isfinite(lengths)) or not all(np.isfinite(angles)):
            return False
        if any(l <= 0 for l in lengths):
            return False
        if any(angle <= 10.0 or angle >= 170.0 for angle in angles):
            return False

        # 检查 distance matrix 是否有效
        dist = struct.distance_matrix
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        if not np.isfinite(min_dist) or min_dist <= 0.1:
            return False

        return True
    except Exception as e:
        print(f"[Invalid Structure] Exception while checking: {e}")
        return False


def main():
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
    # parser.add_argument('--diffusion_epochs', type=int, default=300, help="Number of epochs for diffusion model training")
    # parser.add_argument('--diffusion_batch_size', type=int, default=32, help="Batch size for diffusion model training")
    # parser.add_argument('--save_diffusion_model', type=str, default='diffusion_model.pt', help="Path of diffusion model weights to load for generation")
    # parser.add_argument('--load_diffusion_model', type=str, default='diffusion_model.pt', help="Path of diffusion model weights to load for generation")


    # For generate
    parser.add_argument('--generate_data_csv', type=str, default='data_csv/data_e43V.csv',
                        help="Path to the input data CSV file")
    # parser.add_argument('--load_diffusion_model', type=str, default='diffusion_model.pt', help="Path of fine-tune diffusion model weights to load for generation")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of structures to generate")
    
    # Condition parameters for generation (note: melting_point is log-scale in data)
    parser.add_argument('--cond_melting', type=float, default=np.log(1400.0), help="Desired melting point (log K)")
    parser.add_argument('--cond_density', type=float, default=8.0, help="Desired density (g/cc)")
    parser.add_argument('--cond_form_energy', type=float, default=0.0, help="Desired formation energy (eV/atom)")
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

        # 筛掉超出 MAX_ATOMS 的样本
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)
        print(f"[INFO] Filtered dataset: {len(df)} samples with ≤ {MAX_ATOMS} atoms")

        # 1) 收集出现的所有元素Z
        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums))  # 排序
        print("Distinct elements (by atomic Z) in the dataset:", all_atomic_nums)
        print("Count =", len(all_atomic_nums))

        # 2) 建立 “Z -> 新的类别ID” 的字典
        #    0 依旧留给空位
        z2index = {}
        for i, Z in enumerate(all_atomic_nums, start=1):
            z2index[Z] = i

        print("z2index mapping:", z2index)
        # num_atom_types = len(all_atomic_nums) + 1  # +1 表示空位=0
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
        max_atoms = MAX_ATOMS

        # Initialize models
        decoder = StructureDecoder(
            latent_dim=latent_dim, 
            max_atoms=max_atoms, 
            num_atom_types=num_atom_types
        )
        decoder.index2z = {v: k for k, v in z2index.items()}
        prop_model = PropertyPredictor(latent_dim=latent_dim, cond_dim=cond_dim)
        endecoder = EncoderDecoderModel(
            latent_dim=latent_dim, 
            max_atom_num=max_atoms, 
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
        endecoder, loss_list, class_weights = train_encoder_decoder(
            endecoder, 
            train_dataset, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=1e-4,
            device=device,
            lattice_w=10.0, coord_w=5.0, type_w=10.0, prop_w=0.0,
            patience=20,
            num_atom_types=NUM_ATOM_TYPES,
            save_path=args.save_endecoder
        )
        plot_training_losses(loss_list, "Pretrain-Loss.png")
        
        print(f"[INFO] Loading pretrained weights from: {args.load_endecoder}")
        state_dict = torch.load(args.load_endecoder, map_location=device, weights_only=True)
        endecoder.load_state_dict(state_dict)

        
        class_weights = compute_class_weights(train_dataset, num_atom_types=NUM_ATOM_TYPES).to(device)

        val_metrics = evaluate_encoder_decoder(
            model=endecoder,
            dataset=val_dataset,
            batch_size=args.batch_size,
            device=device, 
            class_weights=class_weights,
            num_atom_types=NUM_ATOM_TYPES,
            lattice_w=10.0, coord_w=5.0, type_w=10.0, prop_w=0.0
        )

        print("Validation results:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        

        # === Train diffusion model ===
        print("[INFO] Training diffusion model ...")
        diff_model = DiffusionModel(
            latent_dim=latent_dim, 
            time_embed_dim=32, 
            hidden_dim=256
        )

        """
        diff_model, diff_loss = train_diffusion_model_unconditional(
            model=diff_model,
            encoder=endecoder.encoder,
            decoder=endecoder.decoder,
            dataset=train_dataset,
            epochs=args.diffusion_epochs,
            batch_size=args.diffusion_batch_size,
            learning_rate=1e-4,
            T=args.t,
            device=device,
            save_path=args.save_diffusion_model,
            lambda_lattice=10.0,
            warmup_epochs=10
        )
                
        plot_diffusion_losses(diff_loss, "Diffusion-Loss.png")
        
        print(f"[INFO] Loading pretrained weights from: {args.load_diffusion_model}")
        state_dict = torch.load(args.load_diffusion_model, map_location=device, weights_only=True)
        diff_model.load_state_dict(state_dict)

        results = evaluate_diffusion_model(
            endecoder.encoder, 
            endecoder.decoder, 
            diff_model, 
            endecoder.prop_pred, 
            val_dataset, 
            T=args.t,
            guidance_scale=args.guidance_scale,
            device=device
        )
        """
        print(f"Training completed.")
    


    elif mode == 'generate':
        ALLOWED_ELEMENTS = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca",
                            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
                            "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", 
                            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", 
                            "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Hf",
                            "Ta", "W", "Re", "Os", "C", "N", "B", "Si"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(args.generate_data_csv)
        df = df.dropna()
        df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)
        print(f"[INFO] Filtered dataset: {len(df)} samples with ≤ {MAX_ATOMS} atoms")

        df["is_valid"] = df["structure"].apply(is_valid_structure)
        df = df[df["is_valid"]].reset_index(drop=True)

        print(f"[INFO] Total valid structures after filtering: {len(df)}")

        df_refer = df[
            (df["melting_point_log"] >= args.cond_melting)
        ].reset_index(drop=True)

        print(f"[INFO] Reference dataset filtered: {len(df_refer)} samples")

        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums))
        z2index = {Z: i+1 for i, Z in enumerate(all_atomic_nums)}
        
        dataset = StructureDataset(
            df, 
            cond_cols=cond_cols, 
            extra_feature_cols=extra_feature_cols, 
            max_atoms=MAX_ATOMS, 
            z2index=z2index
        )

        refer_dataset = StructureDataset(
            df_refer, 
            cond_cols=cond_cols, 
            extra_feature_cols=extra_feature_cols, 
            max_atoms=MAX_ATOMS, 
            z2index=z2index
        )
        
        if torch.cuda.is_available():
            dataset.device = torch.device('cuda')

        latent_dim = 64 + len(extra_feature_cols)
        cond_dim = len(cond_cols)
        max_atoms = MAX_ATOMS
        # num_atom_types = len(dataset.z2index) + 1  # include 0
        num_atom_types = NUM_ATOM_TYPES


        # === Load Encoder-Decoder ===
        print("[INFO] Loading Encoder-Decoder model...")
        endecoder = EncoderDecoderModel(
            atom_fea_dim=64,
            edge_fea_dim=128,
            depth=3,
            max_atom_num=max_atoms,
            extra_fea_dim=len(extra_feature_cols),
            latent_dim=latent_dim,
            num_atom_types=num_atom_types,
            decoder=StructureDecoder(latent_dim=latent_dim, max_atoms=max_atoms, num_atom_types=num_atom_types),
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

        latent_samples = torch.randn(args.num_samples, latent_dim, device=device)
        latent_samples = latent_samples.clone().detach().requires_grad_(True)


        refer_lattices = []
        for i in range(len(refer_dataset)):
            refer_lattices.append(refer_dataset[i]["lattice_label"])
        refer_lattices = torch.stack(refer_lattices).to(device)  # [N, 6]

        N = min(len(df_refer), latent_samples.shape[0])
        assert refer_lattices.shape[0] >= N, "Not enough refer_lattices"
        refer_lattices = refer_lattices[torch.randperm(refer_lattices.shape[0])[: N]]
        

        for i in range(N):
            z_i = latent_samples[i].clone().detach().requires_grad_(True)
            target_lattice_batch = refer_lattices[i]  # [num_samples, 6]
            optimizer = torch.optim.Adam([z_i], lr=1e-2)

            for step in range(200):  # optimization steps
                _, lattice_norm, coords, types_logits = endecoder.decoder(z_i.unsqueeze(0))

                lattice_loss = F.mse_loss(lattice_norm[0], target_lattice_batch)

                dist_loss = 0.0
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

                """
                if step % 5 == 0:
                    print(f"[Opt] Step {step} | Lattice Loss: {lattice_loss.item():.4f} | Dist Loss: {dist_loss.item():.4f}")
                    with torch.no_grad():
                        # 查看全部 latent 优化后的 lattice 是否一致
                        # print(torch.std(lattice_norm[0], dim=0))  # 每个 lattice 维度的标准差
                        lattice_np = lattice_norm[0].detach().cpu().numpy()
                        types = types_logits[0].argmax(dim=-1).cpu().numpy()
                        valid_types = types[types != 0]
                        print(f"[Step {step}] Lattice: {np.round(lattice_np, 3)} | Types: {valid_types}")
                """
            with torch.no_grad():
                _, lattice_norm, coords, types_logits = endecoder.decoder(latent_samples)
            

        A_MIN, A_MAX = 3.0, 18.0
        ALPHA_MIN, ALPHA_MAX = 45.0, 135.0

        # def safe_sigmoid(x):
            # np version sigmoid
        #     return 1.0 / (1.0 + np.exp(-max(min(x, 20.0), -20.0)))
        
        def compute_volume(a, b, c, alpha, beta, gamma):
            alpha, beta, gamma = map(np.radians, [alpha, beta, gamma])
            vol = a * b * c * np.sqrt(
                1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            )
            return vol

        def adjust_to_nearest_valid_lattice(a, b, c, alpha, beta, gamma):
            candidates = []

            # 原始
            vol = compute_volume(a, b, c, alpha, beta, gamma)
            if vol > 5.0:  # 合法就直接用
                return a, b, c, alpha, beta, gamma

            # === 尝试各类晶系修正 ===
            # 1. Cubic
            a_cub = (a + b + c) / 3
            cubic = (a_cub, a_cub, a_cub, 90, 90, 90)

            # 2. Tetragonal
            ab_tet = (a + b) / 2
            tetragonal = (ab_tet, ab_tet, c, 90, 90, 90)

            # 3. Hexagonal
            ab_hex = (a + b) / 2
            hexagonal = (ab_hex, ab_hex, c, 90, 90, 120)

            # 4. Orthorhombic
            orthorhombic = (a, b, c, 90, 90, 90)

            # 5. Trigonal
            trigonal = ((a + b + c)/3,)*3 + ((alpha + beta + gamma)/3,)*3

            # Collect candidates
            candidates = [cubic, tetragonal, hexagonal, orthorhombic, trigonal]

            # Select the one closest to original (Euclidean distance in 6D space)
            original = np.array([a, b, c, alpha, beta, gamma])
            best = None
            best_dist = float('inf')
            for cand in candidates:
                dist = np.linalg.norm(original - np.array(cand))
                vol_cand = compute_volume(*cand)
                if vol_cand > 5.0 and dist < best_dist:
                    best_dist = dist
                    best = cand

            if best is not None:
                return best
            else:
                # fallback to safe orthorhombic
                return 5.0, 5.5, 6.0, 90, 90, 90
            

        def adjust_lattice_to_nearest_valid(lattice_params, 
                                    a_min=3.0, a_max=18.0,
                                    alpha_min=45.0, alpha_max=135.0,
                                    tol=1.0):
            """
            自动将输入的晶格参数调整为最近的符合某种晶系对称性的有效晶格。
            
            输入:
                a, b, c: 晶格长度
                alpha, beta, gamma: 晶格角度
                tol: 判断是否相等的容差，单位为度或 Å

            输出:
                a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj, 晶系名
            """

            a, b, c, alpha, beta, gamma = lattice_params
            
            def is_close(x, y, tol=1.0):
                return abs(x - y) < tol

            def average(*args):
                return sum(args) / len(args)
            
            # 限制角度范围
            alpha = np.clip(alpha, alpha_min, alpha_max)
            beta = np.clip(beta, alpha_min, alpha_max)
            gamma = np.clip(gamma, alpha_min, alpha_max)

            if all(is_close(x, 90.0, tol) for x in [alpha, beta, gamma]):
                if is_close(a, b, tol) and is_close(b, c, tol):
                    a_new = average(a, b, c)
                    return a_new, a_new, a_new, 90.0, 90.0, 90.0, 'cubic'
                elif is_close(a, b, tol):
                    a_new = average(a, b)
                    return a_new, a_new, c, 90.0, 90.0, 90.0, 'tetragonal'
                else:
                    return a, b, c, 90.0, 90.0, 90.0, 'orthorhombic'
            elif is_close(alpha, 90.0, tol) and is_close(gamma, 90.0, tol):
                return a, b, c, 90.0, beta, 90.0, 'monoclinic'
            elif all(is_close(x, 120.0, tol) for x in [alpha, beta, gamma]):
                a_new = average(a, b, c)
                return a_new, a_new, a_new, 120.0, 120.0, 120.0, 'trigonal'
            elif is_close(alpha, 90.0, tol) and is_close(beta, 90.0, tol) and is_close(gamma, 120.0, tol):
                a_new = average(a, b)
                return a_new, a_new, c, 90.0, 90.0, 120.0, 'hexagonal'
            else:
                # 默认 fallback 到 Triclinic：角度取最近的 5°
                alpha_adj = round(alpha / 5) * 5
                beta_adj = round(beta / 5) * 5
                gamma_adj = round(gamma / 5) * 5
                return a, b, c, alpha_adj, beta_adj, gamma_adj, 'triclinic'
    

        def map_lattice_params(lattice_norm_np):
            a_raw, b_raw, c_raw, alpha_raw, beta_raw, gamma_raw = lattice_norm_np
            a = A_MIN + (A_MAX - A_MIN) * a_raw
            b = A_MIN + (A_MAX - A_MIN) * b_raw
            c = A_MIN + (A_MAX - A_MIN) * c_raw
            alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * alpha_raw
            beta  = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * beta_raw
            gamma = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * gamma_raw
            return a, b, c, alpha, beta, gamma
            # return adjust_to_nearest_valid_lattice(a, b, c, alpha, beta, gamma)
        
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
            # print("[SUCCESS] Not pathological.")
            return False

        ###################################################################
        # Main Loop
        # === Start generating structures ===
        ###################################################################

        results = []
        print("[INFO] Start generating structures via latent optimization...")

        NOVELTY_THRESHOLD = 0.2
        for i in range(args.num_samples):
            print(f"--- SAMPLE {i} ---")
            z = latent_samples[i].detach().cpu()
            dist_to_train = ((train_latents_tensor - z)**2).sum(dim=1).sqrt().min()

            if dist_to_train < NOVELTY_THRESHOLD:
                print(f"[Skip] Sample {i} too close to training set (dist = {dist_to_train:.4f})")
                continue


            lattice_params = map_lattice_params(lattice_norm[i].cpu().numpy())
            try:
                a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj, crystal_system = adjust_lattice_to_nearest_valid(lattice_params)
                lattice_params_adj = [a_adj, b_adj, c_adj, alpha_adj, beta_adj, gamma_adj]
                lattice = Lattice.from_parameters(*lattice_params_adj)
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
                    Z = int(types_np[j])
                    el = Element.from_Z(Z)
                    if el.symbol in ALLOWED_ELEMENTS:
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
                if np.min(dist) < 0.3:
                    continue
                num_atoms = len(struct)
                vol = struct.volume
                vol_per_atom = vol / num_atoms
                if vol_per_atom < 3 or vol_per_atom > 50:
                    print("Warning: Unusual volume/atom ratio:", vol_per_atom)
                    continue
                formula = struct.composition.reduced_formula
                struct_json = json.dumps(struct.as_dict())

                unique_structures = []  # 存储用于对比的标准结构
                matcher = StructureMatcher()

                def is_sun_structure(struct, unique_structures):
                    try:
                        # 标准化结构
                        analyzer = SpacegroupAnalyzer(struct, symprec=1e-3)
                        std_struct = analyzer.get_conventional_standard_structure()

                        # 检查是否是有序结构
                        if not std_struct.is_ordered:
                            return False
                        
                        # 检查是否与已有结构重复
                        for s in unique_structures:
                            if matcher.fit(std_struct, s):
                                return False  # 冗余结构
                        
                        unique_structures.append(std_struct)
                        return True

                    except Exception as e:
                        return False
                    
                if is_sun_structure(struct, unique_structures):
                    sga = SpacegroupAnalyzer(struct)
                    spacegroup = sga.get_space_group_symbol()
                    results.append({
                        "formula_pretty": formula, 
                        "structure": struct_json,
                        "spacegroup": spacegroup
                    })

                # Save structure as CIF file
                os.makedirs("generated_CIF", exist_ok=True)
                cif_filename = f"generated_CIF/generated_{i+1}_{formula}.cif"
                try:
                    CifWriter(struct).write_file(cif_filename)
                except Exception as e:
                    print(f"Warning: Failed to write CIF for sample {i+1}: {e}")

                # Save results to CSV
                if not results:
                    print("No structures generated after filtering.")
                else:
                    BEST_MODEL_PATHS = ["best_modelA_mp_finetuned.pth", "best_modelA_rho_finetuned.pth", "best_modelA_fe_finetuned.pth"]
                    TARGET_COLS = ["melting_point_log", "density", "formation_energy_per_atom"]
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    df_new = pd.DataFrame(results)
                    df_new["structure"] = df_new["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
            
                    df_pred = pd.DataFrame()
                    df_pred["formula_pretty"] = df_new["formula_pretty"]
                    df_pred["structure"] = df_new["structure"]

                    def estimate_density_from_structure_and_formula(structure, formula):
                        try:
                            volume = structure.lattice.volume
                            comp = Composition(formula)
                            mass = comp.weight  # 单位 g/mol
                            density = mass / (volume * 1.66054)  # g/cm³
                            return density
                        except Exception as e:
                            print(f"[Warning] Failed to compute density for formula {formula}: {e}")
                            return None

                    # df_pred["estimate_density"] = [
                    #     estimate_density_from_structure_and_formula(struct, formula)
                    #     for struct, formula in zip(df_pred["structure"], df_pred["formula_pretty"])
                    # ]
                    df_pred["estimate_density"] = df_pred["structure"].apply(
                        lambda s: s.density if s is not None else None
                    )
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
                        z2index = {Z: i + 1 for i, Z in enumerate(all_atomic_nums)}  # start from 1

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
                                print(f"❌ Error at row {i} (formula_pretty: {row['formula_pretty']}): {e}")
                                preds.append(np.nan)
                        return preds

                    # Predicting melting_point, density and formation_energy_per_atom
                    for BEST_MODEL_PATH, TARGET_COL in zip(BEST_MODEL_PATHS[:2], TARGET_COLS[:2]):
                        predictions = predict_values(df_new, BEST_MODEL_PATH)
                        if TARGET_COL == "melting_point_log":
                            predictions = np.exp(predictions)
                            df_pred["pred_melting_point"] = predictions
                        else:
                            df_pred["pred_"+TARGET_COL] = predictions


            except Exception as e:
                print(f"[ERROR] Failed to create Structure: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("generated_structures", exist_ok=True)
        output_csv = f"generated_structures/generated_structures_{timestamp}.csv"
        df_pred["structure"] = df_pred["structure"].apply(lambda x: json.dumps(x.as_dict()))
        df_pred.to_csv(output_csv, index=False)
        print(f"Generated {len(results)} structures saved to {output_csv} (CIF files saved for each structure).")
        print("Generation completed.")
        print("--END--")


if __name__ == "__main__":
    main()