import json
import os
import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pymatgen.core import Structure, Lattice as PmgLattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
import copy

# ======================
# CGCNN Feature Extraction Components (与 modelA_v3.py 完全相同)
# ======================

def gaussian_expansion(distance, centers, gamma=40.0):
    return np.exp(-gamma * (distance - centers) ** 2)

def build_cgcnn_graph(structure: Structure, z2index, cutoff=6.0, max_num_nbr=12,
                      radius_step=6.0/127, max_radius=6.0):
    atomic_nums = np.array([z2index[site.specie.Z] for site in structure], dtype=np.int64)
    center_indices, neighbor_indices, _, distances = structure.get_neighbor_list(r=cutoff)
    N = len(structure)
    adjacency = [[] for _ in range(N)]
    dist_list = [[] for _ in range(N)]
    for c_idx, n_idx, dist in zip(center_indices, neighbor_indices, distances):
        adjacency[c_idx].append(n_idx)
        dist_list[c_idx].append(dist)
    num_centers = int(max_radius / radius_step) + 1
    gauss_centers = np.linspace(0, max_radius, num_centers)
    edge_src = []
    edge_dst = []
    edge_features = []
    for i in range(N):
        nbrs = adjacency[i]
        nbr_dists = dist_list[i]
        sorted_nbrs = sorted(zip(nbrs, nbr_dists), key=lambda x: x[1])[:max_num_nbr]
        for j, dist in sorted_nbrs:
            edge_src.append(i)
            edge_dst.append(j)
            edge_fea = gaussian_expansion(dist, gauss_centers)
            edge_features.append(edge_fea)
    if len(edge_features) == 0:
        edge_features = np.zeros((0, num_centers), dtype=np.float32)
    else:
        edge_features = np.stack(edge_features, axis=0).astype(np.float32)
    edge_index = np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
    return atomic_nums, edge_index, edge_features

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor):
    out.index_add_(dim, index, src)
    return out

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int, out: torch.Tensor):
    count = torch.zeros_like(out)
    count.index_add_(dim, index, torch.ones_like(src))
    out.index_add_(dim, index, src)
    out = out / (count + 1e-8)
    return out

class AtomEmbedding(nn.Module):
    def __init__(self, max_atom_num=85, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_atom_num, embedding_dim=embed_dim)
    def forward(self, x: torch.LongTensor):
        return self.embedding(x)

class CGCNNConv(nn.Module):
    def __init__(self, atom_fea_dim: int, edge_fea_dim: int):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_dim + edge_fea_dim, 2 * atom_fea_dim)
    def forward(self, atom_fea: torch.Tensor, edge_index: torch.Tensor, edge_fea: torch.Tensor):
        src, dst = edge_index
        atom_src = atom_fea[src]
        atom_dst = atom_fea[dst]
        edge_in = torch.cat([atom_src, atom_dst, edge_fea], dim=1)
        edge_out = self.fc_full(edge_in)
        gate, core = torch.chunk(edge_out, chunks=2, dim=1)
        gate = torch.sigmoid(gate)
        core = torch.tanh(core)
        message = gate * core
        agg = torch.zeros_like(atom_fea)
        agg = scatter_add(message, dst, dim=0, out=agg)
        new_atom_fea = F.softplus(atom_fea + agg)
        return new_atom_fea

class CGCNNFeatureExtractor(nn.Module):
    def __init__(self, atom_fea_dim=64, edge_fea_dim=128, depth=3,
                 num_atom_types=85, use_extra_fea=True, extra_fea_dim=11):
        super().__init__()
        self.use_extra_fea = use_extra_fea
        self.embed = AtomEmbedding(max_atom_num=num_atom_types, embed_dim=atom_fea_dim)
        self.convs = nn.ModuleList(CGCNNConv(atom_fea_dim, edge_fea_dim) for _ in range(depth))
    def forward(self, node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea=None):
        atom_fea = self.embed(node_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, edge_index, edge_fea)
        num_graphs = crystal_atom_idx.max().item() + 1
        graph_emb = scatter_mean(atom_fea, crystal_atom_idx, dim=0,
                                 out=torch.zeros(num_graphs, atom_fea.shape[1], device=atom_fea.device))
        if self.use_extra_fea and extra_fea is not None:
            graph_emb = torch.cat([graph_emb, extra_fea], dim=1)
        return graph_emb

# ======================
# Dataset (与 modelA_v3.py 相同，只取需要的字段)
# ======================

class StructureDataset(Dataset):
    def __init__(self, df, cond_cols, extra_feature_cols, max_atoms, z2index,
                 a_min=3.0, a_max=18.0, alpha_min=45.0, alpha_max=135.0,
                 cond_mean=None, cond_std=None):
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
        self.extra_mean = self.df[extra_feature_cols].mean()
        self.extra_std = self.df[extra_feature_cols].std().replace({0: 1.0})
        if cond_mean is None:
            self.cond_mean = self.df[cond_cols].mean()
        else:
            self.cond_mean = cond_mean
        if cond_std is None:
            self.cond_std = self.df[cond_cols].std().replace({0: 1.0})
        else:
            self.cond_std = cond_std
        self.device = torch.device('cpu')

    def normalize_lattice_params(self, a, b, c, alpha, beta, gamma):
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
        row = self.df.iloc[idx]
        structure: Structure = row["structure"]
        try:
            analyzer = SpacegroupAnalyzer(structure)
            sg_number = analyzer.get_space_group_number()
            symm_dataset = analyzer.get_symmetry_dataset()
            wyckoff_symbols = symm_dataset.wyckoffs
            counter = Counter(wyckoff_symbols)
            wyckoff_labels = [f"{counter[sym]}{sym}" for sym in wyckoff_symbols]
            wyckoff_species = [site.specie.symbol for site in structure.sites]
            wp_labels_pad = ['0'] * self.max_atoms
            wp_species_pad = ['X'] * self.max_atoms
            for j in range(min(len(wyckoff_labels), self.max_atoms)):
                wp_labels_pad[j] = wyckoff_labels[j]
                wp_species_pad[j] = wyckoff_species[j]
            sg_id = torch.tensor(sg_number, dtype=torch.long)
        except Exception:
            sg_id = torch.tensor(1, dtype=torch.long)
            wp_labels_pad = ['0'] * self.max_atoms
            wp_species_pad = ['X'] * self.max_atoms

        node_fea_np, edge_index_np, edge_fea_np = build_cgcnn_graph(structure, self.z2index)
        node_fea = torch.tensor(node_fea_np, dtype=torch.long)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        edge_fea = torch.tensor(edge_fea_np, dtype=torch.float32)
        N = node_fea.shape[0]
        crystal_atom_idx = torch.zeros(N, dtype=torch.long)

        extra_vals = []
        for col in self.extra_feature_cols:
            val = (row[col] - self.extra_mean[col]) / (self.extra_std[col] + 1e-8)
            extra_vals.append(val)
        extra_vals = torch.tensor(extra_vals, dtype=torch.float32).unsqueeze(0)

        cond_vals = []
        for c in self.cond_cols:
            val = (row[c] - self.cond_mean[c]) / (self.cond_std[c] + 1e-8)
            cond_vals.append(val)
        cond_vals = torch.tensor(cond_vals, dtype=torch.float32)

        latt = structure.lattice
        lattice_norm = self.normalize_lattice_params(latt.a, latt.b, latt.c, latt.alpha, latt.beta, latt.gamma)

        # 元素类型标签（多组分）
        atom_types = [site.specie.Z for site in structure]
        types_pad = np.zeros((self.max_atoms,), dtype=np.int64)
        for j in range(min(len(atom_types), self.max_atoms)):
            Z_j = atom_types[j]
            types_pad[j] = self.z2index.get(Z_j, 0)
        types_label = torch.tensor(types_pad, dtype=torch.long)

        return {
            "node_fea": node_fea,
            "edge_index": edge_index,
            "edge_fea": edge_fea,
            "crystal_atom_idx": crystal_atom_idx,
            "extra_fea": extra_vals,
            "cond": cond_vals,
            "lattice_label": torch.tensor(lattice_norm, dtype=torch.float32),
            "sg_id": sg_id,
            "wyckoff_labels": wp_labels_pad,
            "wyckoff_species": wp_species_pad,
            "types_label": types_label
        }

def collate_fn(batch):
    node_feas = []
    edge_indices = []
    edge_feas = []
    crystal_atom_idxs = []
    extra_feas = []
    conds = []
    lattice_labels = []
    sg_ids = []
    wyckoff_labels_batch = []
    wyckoff_species_batch = []
    types_labels = []
    atom_offset = 0
    for i, data in enumerate(batch):
        n_atoms = data['node_fea'].size(0)
        node_feas.append(data['node_fea'])
        edge_indices.append(data['edge_index'] + atom_offset)
        edge_feas.append(data['edge_fea'])
        crystal_atom_idxs.append(torch.full((n_atoms,), i, dtype=torch.long))
        extra_feas.append(data['extra_fea'].squeeze(0))
        conds.append(data['cond'])
        lattice_labels.append(data['lattice_label'])
        sg_ids.append(data['sg_id'])
        wyckoff_labels_batch.append(data['wyckoff_labels'])
        wyckoff_species_batch.append(data['wyckoff_species'])
        types_labels.append(data['types_label'])
        atom_offset += n_atoms
    return {
        "node_fea": torch.cat(node_feas, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_fea": torch.cat(edge_feas, dim=0),
        "crystal_atom_idx": torch.cat(crystal_atom_idxs, dim=0),
        "extra_fea": torch.stack(extra_feas),
        "cond": torch.stack(conds),
        "lattice_label": torch.stack(lattice_labels),
        "sg_id": torch.stack(sg_ids),
        "wyckoff_labels": wyckoff_labels_batch,
        "wyckoff_species": wyckoff_species_batch,
        "types_label": torch.stack(types_labels)
    }

# ======================
# 新模型：只输出空间群、Wyckoff标签、晶格参数、元素类型
# ======================

class PrototypeDecoder(nn.Module):
    def __init__(self, latent_dim, num_sg=230, num_wyckoff=60, num_elements=44,
                 max_atoms=20, hidden_dim=256):
        super().__init__()
        self.max_atoms = max_atoms
        # 共享 backbone
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 空间群分类头
        self.sg_head = nn.Linear(hidden_dim, num_sg)
        # 晶格参数回归头
        self.lattice_head = nn.Linear(hidden_dim, 6)
        # 元素类型头（每个位点）
        self.type_head = nn.Linear(hidden_dim, max_atoms * num_elements)
        # Wyckoff标签头（每个位点）
        self.wyckoff_head = nn.Linear(hidden_dim, max_atoms * num_wyckoff)

    def forward(self, latent):
        h = self.shared(latent)
        sg_logits = self.sg_head(h)
        lattice_norm = torch.sigmoid(self.lattice_head(h))  # [0,1] 与训练标签一致
        type_logits = self.type_head(h).view(latent.size(0), self.max_atoms, -1)
        wyckoff_logits = self.wyckoff_head(h).view(latent.size(0), self.max_atoms, -1)
        return sg_logits, lattice_norm, type_logits, wyckoff_logits

class EncoderPrototypeModel(nn.Module):
    def __init__(self, atom_fea_dim=64, edge_fea_dim=128, depth=3,
                 num_atom_types=85, extra_fea_dim=11,
                 num_sg=230, num_wyckoff=60, num_elements=44, max_atoms=20,
                 latent_dim=64+11):
        super().__init__()
        self.encoder = CGCNNFeatureExtractor(
            atom_fea_dim=atom_fea_dim, edge_fea_dim=edge_fea_dim, depth=depth,
            num_atom_types=num_atom_types, use_extra_fea=True, extra_fea_dim=extra_fea_dim
        )
        self.decoder = PrototypeDecoder(
            latent_dim=latent_dim, num_sg=num_sg, num_wyckoff=num_wyckoff,
            num_elements=num_elements, max_atoms=max_atoms
        )
    def forward(self, node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea):
        latent = self.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea)
        sg_logits, lattice_norm, type_logits, wyckoff_logits = self.decoder(latent)
        return sg_logits, lattice_norm, type_logits, wyckoff_logits

# ======================
# 损失函数
# ======================

def compute_losses(sg_logits, lattice_pred, type_logits, wyckoff_logits, batch, class_weights, label2index, device):
    # 空间群分类损失
    sg_loss = F.cross_entropy(sg_logits, batch["sg_id"] - 1)  # sg_id 1-230 -> 0-229

    # 晶格参数 MSE
    lattice_loss = F.mse_loss(lattice_pred, batch["lattice_label"])

    # 元素类型损失（忽略填充位点）
    B, N = batch["types_label"].shape
    type_logits_2d = type_logits.view(B * N, -1)
    type_targets = batch["types_label"].view(B * N)
    mask = type_targets > 0
    if mask.sum() > 0:
        type_loss = F.cross_entropy(type_logits_2d[mask], type_targets[mask], weight=class_weights)
    else:
        type_loss = torch.tensor(0.0, device=device)

    # Wyckoff标签损失
    wyckoff_indices = []
    for labels in batch["wyckoff_labels"]:
        indices = [label2index.get(lbl, 0) for lbl in labels]
        wyckoff_indices.append(torch.tensor(indices, dtype=torch.long))
    wyckoff_target = torch.stack(wyckoff_indices).to(device)
    wy_logits_2d = wyckoff_logits.view(B * N, -1)
    wy_targets = wyckoff_target.view(B * N)
    wy_loss = F.cross_entropy(wy_logits_2d, wy_targets)

    return {
        "sg": sg_loss,
        "lattice": lattice_loss,
        "type": type_loss,
        "wyckoff": wy_loss
    }

# ======================
# 训练函数
# ======================

def train_prototype_model(model, train_dataset, val_dataset, label2index, class_weights,
                          epochs=200, batch_size=32, lr=1e-4, device='cpu',
                          save_path='best_prototype_model.pth'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            sg_logits, lattice_pred, type_logits, wyckoff_logits = model(
                batch["node_fea"], batch["edge_index"], batch["edge_fea"],
                batch["crystal_atom_idx"], batch["extra_fea"]
            )
            losses = compute_losses(sg_logits, lattice_pred, type_logits, wyckoff_logits,
                                    batch, class_weights, label2index, device)
            loss = losses["sg"] + losses["lattice"] + losses["type"] + losses["wyckoff"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                sg_logits, lattice_pred, type_logits, wyckoff_logits = model(
                    batch["node_fea"], batch["edge_index"], batch["edge_fea"],
                    batch["crystal_atom_idx"], batch["extra_fea"]
                )
                losses = compute_losses(sg_logits, lattice_pred, type_logits, wyckoff_logits,
                                        batch, class_weights, label2index, device)
                val_loss += (losses["sg"] + losses["lattice"] + losses["type"] + losses["wyckoff"]).item()

        print(f"Epoch {epoch}: train loss={total_loss/len(train_loader):.4f}, val loss={val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model

# ======================
# 主程序
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'])
    parser.add_argument('--data_csv', type=str, default='data_csv/data_e43V.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_model', type=str, default='prototype_model.pth')
    parser.add_argument('--load_model', type=str, default='prototype_model.pth')
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    cond_cols = ["melting_point_log", "density"]
    extra_feature_cols = ["num_of_atoms", "energy_above_hull", "band_gap", "charge",
                           "electronic_energy", "total_enthalpy", "total_entropy",
                           "dielectric_constant", "refractive_index", "stoichiometry_sum", "volume_per_atom"]
    MAX_ATOMS = 20
    NUM_ELEMENTS = 44
    ALLOWED_ELEMENTS = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca",
                        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
                        "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr",
                        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
                        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Hf",
                        "Ta", "W", "Re", "Os", "C", "N", "B", "Si"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        df = pd.read_csv(args.data_csv)
        df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
        # 过滤
        allowed_Z = set(Element(sym).Z for sym in ALLOWED_ELEMENTS)
        df = df[df["structure"].apply(lambda s: all(site.specie.Z in allowed_Z for site in s))]
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)
        print(f"Training samples: {len(df)}")

        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums))
        z2index = {Z: i+1 for i, Z in enumerate(all_atomic_nums)}

        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = StructureDataset(df_train, cond_cols, extra_feature_cols, MAX_ATOMS, z2index)
        val_dataset = StructureDataset(df_val, cond_cols, extra_feature_cols, MAX_ATOMS, z2index,
                                       cond_mean=train_dataset.cond_mean, cond_std=train_dataset.cond_std)

        # 建立 Wyckoff 词汇表
        all_labels = set()
        for i in range(len(train_dataset)):
            for lbl in train_dataset[i]["wyckoff_labels"]:
                if lbl != '0':
                    all_labels.add(lbl)
        unique_labels = sorted(all_labels)
        label2index = {'0': 0}
        for i, lbl in enumerate(unique_labels, start=1):
            label2index[lbl] = i
        print(f"Wyckoff vocabulary size: {len(label2index)}")

        # 类别权重（只用于元素类型）
        all_types = []
        for i in range(len(train_dataset)):
            all_types.append(train_dataset[i]["types_label"].numpy())
        all_types = np.concatenate(all_types)
        type_counts = np.bincount(all_types, minlength=NUM_ELEMENTS+1)
        class_weights = 1.0 / (type_counts + 1e-6)
        class_weights[0] = 0.1  # 降低填充类的权重
        class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)

        model = EncoderPrototypeModel(
            num_atom_types=NUM_ELEMENTS+1, extra_fea_dim=len(extra_feature_cols),
            num_sg=230, num_wyckoff=len(label2index), num_elements=NUM_ELEMENTS+1,
            max_atoms=MAX_ATOMS
        )
        train_prototype_model(model, train_dataset, val_dataset, label2index, class_weights,
                              epochs=args.epochs, batch_size=args.batch_size, device=device,
                              save_path=args.save_model)
        print("Training completed.")

    elif args.mode == 'generate':
        df = pd.read_csv(args.data_csv)
        df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
        allowed_Z = set(Element(sym).Z for sym in ALLOWED_ELEMENTS)
        df = df[df["structure"].apply(lambda s: all(site.specie.Z in allowed_Z for site in s))]
        df = df[df["num_of_atoms"] <= MAX_ATOMS].reset_index(drop=True)

        all_atomic_nums = set()
        for struct in df["structure"]:
            for site in struct:
                all_atomic_nums.add(site.specie.Z)
        all_atomic_nums = sorted(list(all_atomic_nums))
        z2index = {Z: i+1 for i, Z in enumerate(all_atomic_nums)}
        index2z = {i: Z for Z, i in z2index.items()}

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = StructureDataset(train_df, cond_cols, extra_feature_cols, MAX_ATOMS, z2index)
        val_dataset = StructureDataset(val_df, cond_cols, extra_feature_cols, MAX_ATOMS, z2index,
                                       cond_mean=train_dataset.cond_mean, cond_std=train_dataset.cond_std)

        all_labels = set()
        for i in range(len(train_dataset)):
            for lbl in train_dataset[i]["wyckoff_labels"]:
                if lbl != '0':
                    all_labels.add(lbl)
        unique_labels = sorted(all_labels)
        label2index = {'0': 0}
        for i, lbl in enumerate(unique_labels, start=1):
            label2index[lbl] = i
        index2label = {i: lbl for lbl, i in label2index.items()}

        model = EncoderPrototypeModel(
            num_atom_types=NUM_ELEMENTS+1, extra_fea_dim=len(extra_feature_cols),
            num_sg=230, num_wyckoff=len(label2index), num_elements=NUM_ELEMENTS+1,
            max_atoms=MAX_ATOMS
        )
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        model.to(device)
        model.eval()

        # ===== 计算训练集潜在向量用于 novelty filter =====
        train_latents = []
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                latent = model.encoder(
                    batch["node_fea"], batch["edge_index"], batch["edge_fea"],
                    batch["crystal_atom_idx"], batch["extra_fea"]
                )
                train_latents.append(latent.cpu().numpy())
        train_latents = np.concatenate(train_latents, axis=0)

        # ===== 筛选参考数据集（高熔点+低密度）=====
        cond_melting = np.log(1400.0)   # 默认 target melting point (log scale)
        cond_density = 8.0              # 默认 target density
        df_refer = df[
            (df["melting_point_log"] >= cond_melting) &
            (df["density"] <= cond_density)
        ].reset_index(drop=True)
        print(f"[INFO] Reference dataset for latent optimization: {len(df_refer)} samples")

        refer_dataset = StructureDataset(
            df_refer, cond_cols, extra_feature_cols, MAX_ATOMS, z2index,
            cond_mean=train_dataset.cond_mean, cond_std=train_dataset.cond_std
        )

        # 提取参考数据集的晶格参数标签
        refer_lattice_labels = []
        for i in range(len(refer_dataset)):
            refer_lattice_labels.append(refer_dataset[i]["lattice_label"])
        refer_lattice_labels = torch.stack(refer_lattice_labels).to(device)

        latent_dim = train_latents.shape[1]

        # ===== 条件生成 =====
        generated = []
        for i in range(args.num_samples):
            z = torch.randn(1, latent_dim, device=device)

            # 潜在向量优化（full_guided）
            if len(refer_lattice_labels) > 0:
                target_lattice = refer_lattice_labels[torch.randint(0, len(refer_lattice_labels), (1,))].to(device)
                z_opt = z.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([z_opt], lr=1e-2)

                for step in range(200):
                    _, lattice_pred, _, _ = model.decoder(z_opt)
                    lattice_loss = F.mse_loss(lattice_pred, target_lattice)
                    loss = lattice_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                z = z_opt.detach()

            # novelty 检查
            dist = np.min(np.linalg.norm(train_latents - z.cpu().numpy(), axis=1))
            if dist < 0.2:
                continue

            with torch.no_grad():
                sg_logits, lattice_norm, type_logits, wyckoff_logits = model.decoder(z)

            sg_pred = torch.argmax(sg_logits, dim=1).item() + 1
            lattice_np = lattice_norm.cpu().numpy()[0]
            a = lattice_np[0] * (18-3) + 3
            b = lattice_np[1] * (18-3) + 3
            c = lattice_np[2] * (18-3) + 3
            alpha = lattice_np[3] * (135-45) + 45
            beta  = lattice_np[4] * (135-45) + 45
            gamma = lattice_np[5] * (135-45) + 45

            # 检查晶格合法性
            try:
                lattice = PmgLattice.from_parameters(a, b, c, alpha, beta, gamma)
                vol = lattice.volume
                if vol < 5.0 or not math.isfinite(vol):
                    continue
            except Exception:
                continue

            type_pred = torch.argmax(type_logits, dim=2).cpu().numpy()[0]
            wy_pred = torch.argmax(wyckoff_logits, dim=2).cpu().numpy()[0]

            # 解码元素类型
            elements = []
            for t in type_pred:
                if t != 0 and t in index2z:
                    elements.append(Element.from_Z(index2z[t]).symbol)
            if len(set(elements)) < 2 or len(elements) > MAX_ATOMS:
                continue
            if len(set(elements)) > 6:
                continue

            wy_labels = []
            for w in wy_pred:
                if w != 0:
                    wy_labels.append(index2label.get(w, '0'))
            wy_labels = wy_labels[:len(elements)]

            generated.append({
                "sg": sg_pred,
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "elements": elements,
                "wyckoff_labels": wy_labels
            })

        result_df = pd.DataFrame(generated)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_prototypes_{timestamp}.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Generated {len(result_df)} prototypes, saved to {output_path}")