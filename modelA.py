import pandas as pd
import numpy as np
import json
from pymatgen.core import Structure
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
import ast
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import argparse

def gaussian_expansion(distance, centers, gamma=40.0):
    """
    distance: scalar or array
    centers: array of shape [M], e.g. np.linspace(0,6,50)
    gamma: float, default=40
    return: array of shape [M]
    """
    return np.exp(-gamma * (distance - centers)**2)

def build_cgcnn_graph(structure: Structure, 
                      cutoff=6.0, 
                      max_num_nbr=12,
                      radius_step=6.0/127,
                      max_radius=6.0):
    """
    构建 CGCNN 风格的晶体图，返回:
      - node_fea: [N]，即 atomic numbers（int 类型）
      - edge_index: [2, E]，即边连接关系
      - edge_fea: [E, edge_dim]，即高斯展开距离
    """
    # 1. 节点特征：原子序号 (int)
    atomic_nums = np.array([site.specie.Z for site in structure], dtype=np.int64)  # shape [N]

    # 2. 邻居查找
    center_indices, neighbor_indices, images, distances = structure.get_neighbor_list(r=cutoff)

    N = len(structure)
    adjacency = [[] for _ in range(N)]
    dist_list = [[] for _ in range(N)]

    for c_idx, n_idx, dist_ij in zip(center_indices, neighbor_indices, distances):
        adjacency[c_idx].append(n_idx)
        dist_list[c_idx].append(dist_ij)

    edge_src = []
    edge_dst = []
    edge_fea_list = []

    # 3. 准备高斯展开中心
    num_centers = int(max_radius / radius_step) + 1
    gauss_centers = np.linspace(0, max_radius, num_centers)

    # 4. 构建边及其特征
    for i in range(N):
        nbrs = adjacency[i]
        nbr_dists = dist_list[i]
        sorted_pairs = sorted(zip(nbrs, nbr_dists), key=lambda x: x[1])
        sorted_pairs = sorted_pairs[:max_num_nbr]
        
        for (j, dist_ij) in sorted_pairs:
            edge_src.append(i)
            edge_dst.append(j)
            edge_fea_i = gaussian_expansion(dist_ij, gauss_centers)
            edge_fea_list.append(edge_fea_i)

    edge_index = np.stack([edge_src, edge_dst], axis=0)  # shape [2, E]
    edge_fea   = np.stack(edge_fea_list, axis=0)         # shape [E, edge_dim]

    # node_fea 为原子序号向量，形状 [N]，类型 int64
    return atomic_nums, edge_index, edge_fea

class CGCNNStructureDataset(Dataset):
    def __init__(self, 
                 df, 
                 target_col="melting_point_log",
                 cutoff=6.0, 
                 max_num_nbr=12, 
                 radius_step=6.0/127, 
                 max_radius=6.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

        self.extra_feature_cols = [
            "num_of_atoms", "energy_above_hull", "band_gap", "charge",
            "electronic_energy", "total_enthalpy", "total_entropy",
            "dielectric_constant", "refractive_index", "stoichiometry_sum", "volume_per_atom"
        ]

        # 计算 extra_fea 的 mean/std
        self.extra_mean = []
        self.extra_std = []
        for col in self.extra_feature_cols:
            mean = df[col].mean()
            std = df[col].std() if df[col].std() != 0 else 1.0
            self.extra_mean.append(mean)
            self.extra_std.append(std)

        self.extra_mean_t = torch.tensor(self.extra_mean, dtype=torch.float32)
        self.extra_std_t  = torch.tensor(self.extra_std,  dtype=torch.float32)

        self.cutoff = cutoff
        self.max_num_nbr = max_num_nbr
        self.radius_step = radius_step
        self.max_radius = max_radius

    def get_extra_mean_std(self):
        return self.extra_mean_t, self.extra_std_t

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        structure = row["structure"]

        # 构建图（atomic number -> [N], int64）
        node_fea, edge_index, edge_fea = build_cgcnn_graph(
            structure,
            cutoff=self.cutoff,
            max_num_nbr=self.max_num_nbr,
            radius_step=self.radius_step,
            max_radius=self.max_radius
        )
        node_fea = torch.tensor(node_fea, dtype=torch.long)         # [N]
        edge_index = torch.tensor(edge_index, dtype=torch.long)     # [2, E]
        edge_fea = torch.tensor(edge_fea, dtype=torch.float32)      # [E, M]

        # 目标值（只预测一个值，不标准化）
        y_val = torch.tensor([row[self.target_col]], dtype=torch.float32)  # shape [1]

        # 标准化 extra_fea
        extra_fea_vals = []
        for col, mean, std in zip(self.extra_feature_cols, self.extra_mean, self.extra_std):
            val = row[col]
            val_norm = (val - mean) / (std + 1e-8)
            extra_fea_vals.append(val_norm)
        extra_fea = torch.tensor(extra_fea_vals, dtype=torch.float32)  # shape [extra_dim]

        # crystal_atom_idx
        N = node_fea.shape[0]
        crystal_atom_idx = torch.zeros(N, dtype=torch.long)  # 所有属于同一个晶体

        return {
            "node_fea": node_fea,               # [N] -> int64
            "edge_index": edge_index,           # [2, E]
            "edge_fea": edge_fea,               # [E, M]
            "crystal_atom_idx": crystal_atom_idx,  # [N]
            "y": y_val,                         # [1]
            "extra_fea": extra_fea              # [extra_dim]
        }
    
def scatter_add(src, index, dim, out):
    out.index_add_(dim, index, src)
    return out


def scatter_mean(src, index, dim, out):
    count = torch.zeros_like(out)
    count.index_add_(dim, index, torch.ones_like(src)) 
    out.index_add_(dim, index, src) 
    out = out / (count + 1e-8) 
    return out


class AtomEmbedding(nn.Module):
    """
    把 (atomic_number) => embedding 向量
    """
    def __init__(self, max_atom_num=100, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(max_atom_num, embed_dim)
        # random init
    def forward(self, x):
        """
        x: shape [N], atomic_number
        return: shape [N, embed_dim]
        """
        return self.embedding(x)
    
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
                 max_atom_num=100,
                 depth=3, 
                 use_extra_fea=False,
                 extra_fea_dim=10):
        super().__init__()
        self.use_extra_fea = use_extra_fea   
        self.extra_fea_dim = extra_fea_dim

        self.embed = AtomEmbedding(max_atom_num, atom_fea_dim)

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


def collate_cgcnn(batch_list):
    """
    把单个样本 list 拼接成 batch 图，供模型一次前向传播。
    每个样本的结构：
      - node_fea: [N]
      - edge_index: [2, E]
      - edge_fea: [E, edge_dim]
      - crystal_atom_idx: [N]
      - y: [1]
      - extra_fea: [extra_dim]
    """
    node_feas = []
    edge_indexes = []
    edge_feas = []
    crystal_atom_idx = []
    ys = []
    extra_feas = []

    n_offset = 0
    for i, sample in enumerate(batch_list):
        n = sample["node_fea"].size(0)
        e = sample["edge_index"].size(1)

        node_feas.append(sample["node_fea"])  # [n]
        ei = sample["edge_index"] + n_offset
        edge_indexes.append(ei)
        edge_feas.append(sample["edge_fea"])  # [e, edge_dim]

        cat_idx = torch.full((n,), i, dtype=torch.long)
        crystal_atom_idx.append(cat_idx)

        ys.append(sample["y"])  # [1]
        extra_feas.append(sample["extra_fea"].unsqueeze(0))  # [1, extra_dim]

        n_offset += n

    node_feas = torch.cat(node_feas, dim=0)       # [sum_n]
    edge_indexes = torch.cat(edge_indexes, dim=1) # [2, sum_e]
    edge_feas = torch.cat(edge_feas, dim=0)       # [sum_e, edge_dim]
    crystal_atom_idx = torch.cat(crystal_atom_idx, dim=0)  # [sum_n]
    ys = torch.cat(ys, dim=0).unsqueeze(1)         # [B] -> [B, 1]
    extra_feas = torch.cat(extra_feas, dim=0)      # [B, extra_dim]

    return {
        "node_fea": node_feas,
        "edge_index": edge_indexes,
        "edge_fea": edge_feas,
        "crystal_atom_idx": crystal_atom_idx,
        "y": ys,
        "extra_fea": extra_feas
    }


def predict_test_set(model, test_loader, target_col, output_csv="test_predictions.csv"):
    """
    针对单输出预测熔点的模型，评估测试集性能。
    要求模型输出为 log(melting_point + 1)，需要取 exp() - 1 还原。
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_data in test_loader:
            node_fea = batch_data["node_fea"].to(device)             # [N]
            edge_index = batch_data["edge_index"].to(device)         # [2, E]
            edge_fea = batch_data["edge_fea"].to(device)             # [E, edge_dim]
            crystal_idx = batch_data["crystal_atom_idx"].to(device)  # [N]
            y_true = batch_data["y"].to(device)                      # [B, 1]
            extra_fea = batch_data["extra_fea"].to(device)           # [B, extra_dim]

            # 模型预测：输出为 log(melting_point + 1)
            y_pred = model(node_fea, edge_index, edge_fea, crystal_idx, extra_fea=extra_fea)  # [B, 1]

            if target_col == "melting_point_log":
                # 还原真实熔点值
                pred = torch.exp(y_pred) - 1.0   # [B, 1]
                true = torch.exp(y_true) - 1.0   # [B, 1]
            else:
                pred = y_pred.clone()
                true = y_true.clone()

            predictions.extend(pred.squeeze(1).cpu().tolist())  # 转为 [B]
            true_values.extend(true.squeeze(1).cpu().tolist())  # 转为 [B]


    df_results = pd.DataFrame({
        "pred": predictions,
        "true": true_values
    })


    mae = mean_absolute_error(df_results["true"], df_results["pred"])
    r2 = r2_score(df_results["true"], df_results["pred"])
    print(f"📈 MAE: {mae:.2f}")
    print(f"📈 R² : {r2:.4f}")


    df_results.to_csv(output_csv, index=False)
    print(f"✅ Prediction results saved to: {output_csv}")

    return df_results


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    # 这是一种介于 L1 和 L2 之间的损失函数，又叫 Huber Loss
    # 对小误差行为像 MSE（收敛快）
	# 对大误差行为像 MAE（鲁棒）

    for batch_data in loader:
        node_fea = batch_data["node_fea"].to(device)
        edge_index = batch_data["edge_index"].to(device)
        edge_fea = batch_data["edge_fea"].to(device)
        crystal_idx = batch_data["crystal_atom_idx"].to(device)
        y = batch_data["y"].to(device)
        extra_fea = batch_data["extra_fea"].to(device)

        pred = model(node_fea, edge_index, edge_fea, crystal_idx, extra_fea=extra_fea)
        loss = criterion(pred, y)  # MSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss() 

    with torch.no_grad():
        for batch_data in loader:
            node_fea = batch_data["node_fea"].to(device)
            edge_index = batch_data["edge_index"].to(device)
            edge_fea = batch_data["edge_fea"].to(device)
            crystal_idx = batch_data["crystal_atom_idx"].to(device)
            y = batch_data["y"].to(device)
            extra_fea = batch_data["extra_fea"].to(device)

            pred = model(node_fea, edge_index, edge_fea, crystal_idx, extra_fea=extra_fea)
            loss = criterion(pred, y)
            total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def stratified_split(df, target_col, batch_size, seed):
    df[target_col+"_bin"] = pd.qcut(df[target_col], q=10, labels=False, duplicates='drop')

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, df[target_col+"_bin"]))

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_bins = df[target_col+"_bin"].iloc[temp_idx].reset_index(drop=True)

    splitter_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(splitter_val_test.split(temp_df, temp_bins))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = temp_df.iloc[val_idx].reset_index(drop=True)
    df_test  = temp_df.iloc[test_idx].reset_index(drop=True)

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    plt.figure(figsize=(5, 5), dpi=300)
    plt.hist(df_train[target_col], bins=30, alpha=0.5, label="Train")
    plt.hist(df_val[target_col], bins=30, alpha=0.5, label="Val")
    plt.hist(df_test[target_col], bins=30, alpha=0.5, label="Test")
    plt.legend()
    plt.title("Variance Value Distribution in Each Split")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.savefig("Variance_Value_Distribution_in_Each_Split.png", dpi=300, bbox_inches="tight")
    # plt.show()

    train_ds = CGCNNStructureDataset(df_train, target_col=target_col)
    val_ds   = CGCNNStructureDataset(df_val, target_col=target_col)
    test_ds  = CGCNNStructureDataset(df_test, target_col=target_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_cgcnn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_cgcnn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_cgcnn)

    return train_loader, val_loader, test_loader


def main_training_example(
        df, 
        TARGET_COL,
        EPOCH_NUM, 
        BATCH_SIZE, 
        PATIENCE,
        LEARNING_RATE,
        BEST_MODEL_PATH, 
        TEST_PREDICTIONS_CSV, 
        USE_EXTRA_FEA,
        EXTRA_FEA_DIM,
        seed=42
    ):

    train_loader, val_loader, test_loader = stratified_split(df, target_col=TARGET_COL, batch_size=BATCH_SIZE, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = CGCNN(atom_fea_dim=64, edge_fea_dim=128, num_targets=1, max_atom_num=100, depth=3, use_extra_fea=USE_EXTRA_FEA, extra_fea_dim=EXTRA_FEA_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stop_counter = 0

    for epoch in range(1, EPOCH_NUM + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved at epoch {epoch} with val_loss={val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        if early_stop_counter == 0:
            _ = predict_test_set(model, val_loader,TARGET_COL, output_csv=f"val_epoch{epoch}_pred.csv")

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss = evaluate(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}")

    df_predictions = predict_test_set(model, test_loader, TARGET_COL, output_csv=TEST_PREDICTIONS_CSV)

    return model, df_predictions, train_losses, val_losses


def filter_by_element_overlap(df_all, allowed_elements, threshold=0.5):
    def is_allowed_fraction_ok(row_elements):
        try:
            # 是字符串，先转换成真正的列表
            if isinstance(row_elements, str):
                row_elements = ast.literal_eval(row_elements)
        except Exception:
            return False  # 解析失败直接过滤
        
        if not row_elements:
            return False
        
        allowed_count = sum(e in allowed_elements for e in row_elements)
        return allowed_count / len(row_elements) >= threshold

    mask = df_all["elements_new"].apply(is_allowed_fraction_ok)
    filtered_df = df_all[mask].reset_index(drop=True)
    return filtered_df

def plot_loss_curve(train_losses, val_losses, FIGURE_PATH):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    # plt.show()


def main_transfer_training_example(
        df, 
        TARGET_COL,
        pretrained_model_path, 
        EPOCH_NUM, 
        BATCH_SIZE, 
        PATIENCE,
        LEARNING_RATE,
        BEST_MODEL_PATH, 
        TEST_PREDICTIONS_CSV, 
        USE_EXTRA_FEA,
        EXTRA_FEA_DIM,
        seed=42,
        freeze_partially=False
    ):

    
    train_loader, val_loader, test_loader = stratified_split(df, target_col=TARGET_COL, batch_size=BATCH_SIZE, seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CGCNN(atom_fea_dim=64, edge_fea_dim=128, num_targets=1, max_atom_num=100, depth=3, use_extra_fea=USE_EXTRA_FEA, extra_fea_dim=EXTRA_FEA_DIM)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.to(device)
    print(f"✅ Loaded pretrained CGCNN model from {pretrained_model_path}")

    # 可选冻结部分参数（只微调输出层
    if freeze_partially:
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.fc_out.parameters():
            param.requires_grad = True
        print("🔒 Froze all layers except fc_out (only finetuning output layer).")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stop_counter = 0

    # 开始迁移训练
    for epoch in range(1, EPOCH_NUM + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"[Transfer] Epoch {epoch:03d}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Best finetuned model saved at epoch {epoch} (val_loss = {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement. ⏸ Early stop counter: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("⛔️ Early stopping triggered.")
                break

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    test_loss = evaluate(model, test_loader, device)
    print(f"[Transfer] Final Test Loss: {test_loss:.4f}")

    df_predictions = predict_test_set(model, test_loader, TARGET_COL, output_csv=TEST_PREDICTIONS_CSV)

    return model, df_predictions, train_losses, val_losses


# =============================================== #


def get_args():
    parser = argparse.ArgumentParser(description="Training script for CGCNN model.")

    parser.add_argument('--target_col', type=str, default='melting_point_log',
                        help='Name of the target column to predict')
    parser.add_argument('--abrev', type=str, default='mp',
                        help='Abbreviation of the property (used in saving logs/models etc.)')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    TARGET_COL = args.target_col
    ABREV = args.abrev

    print(f"Using target column: {TARGET_COL}")
    print(f"Using abbreviation: {ABREV}")

    DATA_CSV = "data_csv/data_all.csv"

    df_all = pd.read_csv(DATA_CSV)
    df_all = df_all.dropna()
    print(f"删除NaN后剩余 {len(df_all)} 条数据")
    # 解析 JSON 并转换为 pymatgen.Structure
    df_all["structure"] = df_all["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))

    ALLOWED_ELEMENTS = set([
        "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe",
        "Co","Ni","Cu","Zn","Ga","Ge","Rb","Sr","Y","Zr","Nb","Mo","Tc",
        "Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Hf","Ta","W","Re","Os",
        "C","N","B","Si"
    ])

    df_filt = filter_by_element_overlap(df_all, ALLOWED_ELEMENTS, threshold=0.5)
    print(f"满足Threshod > 0.5 的数据条数: {len(df_filt)}")
    df_filt["volume_per_atom"] = df_filt["structure"].apply(lambda s: s.volume / len(s))


    EPOCH_NUM = 200
    BATCH_SIZE = 16
    PATIENCE = 20
    LEARNING_RATE = 1e-4


    BEST_MODEL_PATH = f"best_modelA_{ABREV}_all.pth"
    TEST_PREDICTIONS_CSV = f"modelA_test_predictions_{ABREV}_all.csv"

    model, df_predictions, train_losses, val_losses = main_training_example(
        df_filt, 
        TARGET_COL,
        EPOCH_NUM, 
        BATCH_SIZE, 
        PATIENCE,
        LEARNING_RATE,
        BEST_MODEL_PATH, 
        TEST_PREDICTIONS_CSV,
        False, # with no extras
        0
    )

    BEST_MODEL_PATH_EX = f"best_modelA_{ABREV}_all_extras.pth"
    TEST_PREDICTIONS_CSV_EX = f"modelA_test_predictions_{ABREV}_all_extras.csv"

    model_ex, df_predictions_ex, train_losses_ex, val_losses_ex = main_training_example(
        df_filt, 
        TARGET_COL,
        EPOCH_NUM, 
        BATCH_SIZE, 
        PATIENCE,
        LEARNING_RATE,
        BEST_MODEL_PATH_EX, 
        TEST_PREDICTIONS_CSV_EX,
        True, # with extras
        11
    )

    plot_loss_curve(train_losses, val_losses, f"modelA_{ABREV}_loss.png")
    plot_loss_curve(train_losses_ex, val_losses_ex, f"modelA_{ABREV}_loss_extras.png")


    df_allowed = pd.read_csv("data_csv/data_e43V.csv")
    df_allowed = df_allowed.dropna()
    print(f"删除NaN后剩余 {len(df_allowed)} 条数据")
    df_allowed["structure"] = df_allowed["structure"].apply(lambda x: Structure.from_dict(json.loads(x)))
    df_allowed["volume_per_atom"] = df_allowed["structure"].apply(lambda s: s.volume / len(s))

    EPOCH_NUM_FINETUINED = 100
    BATCH_SIZE_FINETUINED = 8
    PATIENCE_FINETUINED = 20
    LEARNING_RATE_FINETUINED = 5e-5

    BEST_FINETUINED_MODEL_PATH = f"best_modelA_{ABREV}_finetuned.pth"
    TEST_PREDICTIONS_CSV_FINETUINED = f"modelA_test_predictions_{ABREV}_finetuned.csv"
    model, df_predictions, train_losses, val_losses = main_transfer_training_example(
        df_allowed,
        TARGET_COL,
        BEST_MODEL_PATH,
        EPOCH_NUM_FINETUINED,
        BATCH_SIZE_FINETUINED,
        PATIENCE_FINETUINED,
        LEARNING_RATE_FINETUINED,
        BEST_FINETUINED_MODEL_PATH,
        TEST_PREDICTIONS_CSV_FINETUINED,
        False,
        0,
        freeze_partially=False
    )

    BEST_FINETUINED_MODEL_PATH_EX = f"best_modelA_{ABREV}_finetuned_extras.pth"
    TEST_PREDICTIONS_CSV_FINETUINED_EX = f"modelA_test_predictions_{ABREV}_finetuned_extras.csv"
    model_ex, df_predictions_ex, train_losses_ex, val_losses_ex = main_transfer_training_example(
        df_allowed,
        TARGET_COL,
        BEST_MODEL_PATH,
        EPOCH_NUM_FINETUINED,
        BATCH_SIZE_FINETUINED,
        PATIENCE_FINETUINED,
        LEARNING_RATE_FINETUINED,
        BEST_FINETUINED_MODEL_PATH_EX,
        TEST_PREDICTIONS_CSV_FINETUINED_EX,
        True,
        11,
        freeze_partially=False
    )

    plot_loss_curve(train_losses, val_losses, f"modelA_{ABREV}_finetuned_loss.png")
    plot_loss_curve(train_losses_ex, val_losses_ex, f"modelA_{ABREV}_finetuned_loss_extras.png")


if __name__ == '__main__':
    main()