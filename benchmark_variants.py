import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Structure, Lattice as PmgLattice
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.model_selection import train_test_split


COND_COLS = ["melting_point_log", "density", "formation_energy_per_atom"]
EXTRA_FEATURE_COLS = [
    "num_of_atoms", "energy_above_hull", "band_gap", "charge",
    "electronic_energy", "total_enthalpy", "total_entropy",
    "dielectric_constant", "refractive_index", "stoichiometry_sum", "volume_per_atom"
]
MAX_ATOMS = 20
NUM_ATOM_TYPES = 44
ALLOWED_ELEMENTS = [
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Hf", "Ta", "W", "Re", "Os", "C", "N", "B", "Si"
]


def load_model_module(module_path: str):
    spec = importlib.util.spec_from_file_location("modelA_v3", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def filter_dataframe(df: pd.DataFrame, model_mod, max_atoms: int, allowed_elements: list[str]) -> pd.DataFrame:
    df = df.dropna().copy()
    df["structure"] = df["structure"].apply(lambda x: Structure.from_dict(json.loads(x)) if isinstance(x, str) else x)
    df = df[df["num_of_atoms"] <= max_atoms].reset_index(drop=True)

    allowed_Z = set(Element(sym).Z for sym in allowed_elements)

    def _all_sites_allowed(struct: Structure):
        try:
            return all(site.specie.Z in allowed_Z for site in struct)
        except Exception:
            return False

    df = df[df["structure"].apply(_all_sites_allowed)].reset_index(drop=True)
    df["is_valid"] = df["structure"].apply(model_mod.is_valid_structure)
    df = df[df["is_valid"]].reset_index(drop=True)
    return df


def build_z2index(df: pd.DataFrame) -> Dict[int, int]:
    atomic_nums = sorted({site.specie.Z for struct in df["structure"] for site in struct})
    return {z: i + 1 for i, z in enumerate(atomic_nums)}


def build_label2index(dataset) -> Dict[str, int]:
    all_labels = set()
    for idx in range(len(dataset)):
        all_labels.update(dataset[idx]["wyckoff_labels"])
    unique_labels = sorted(all_labels - {"0"})
    label2index = {"0": 0}
    for i, label in enumerate(unique_labels, start=1):
        label2index[label] = i
    return label2index


def partial_load_matching_weights(model: torch.nn.Module, state_dict: dict) -> Tuple[list[str], list[str]]:
    model_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped


class EncoderDecoderUnconditional(torch.nn.Module):
    """Benchmark-only unconditional variant.

    It preserves the decoder architecture but removes `extra_fea` from the encoder output.
    This model should be treated as a benchmark-only branch and trained separately.
    """
    def __init__(self, model_mod, atom_fea_dim=64, edge_fea_dim=128, depth=3, max_atom_num=20,
                 num_atom_types=85, decoder=None, prop_pred=None):
        super().__init__()
        self.encoder = model_mod.CGCNNFeatureExtractor(
            atom_fea_dim=atom_fea_dim,
            edge_fea_dim=edge_fea_dim,
            depth=depth,
            num_atom_types=num_atom_types,
            use_extra_fea=False,
            extra_fea_dim=0,
        )
        self.decoder = decoder
        self.prop_pred = prop_pred

    def forward(self, node_fea, edge_index, edge_fea, crystal_atom_idx, extra_fea=None):
        latent = self.encoder(node_fea, edge_index, edge_fea, crystal_atom_idx, None)
        lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params = self.decoder(latent)
        prop_out = None
        if self.prop_pred is not None:
            prop_out = self.prop_pred(latent)
        return lattice, lattice_norm, coords, types_logits, pred_wp_logits, pred_wp_params, prop_out


def build_full_model(model_mod, z2index, label2index, device):
    latent_dim = 64 + len(EXTRA_FEATURE_COLS)
    decoder = model_mod.StructureDecoder(
        latent_dim=latent_dim,
        max_atoms=MAX_ATOMS,
        num_atom_types=NUM_ATOM_TYPES,
        z2index=z2index,
        num_wp_labels=len(label2index),
        allowed_elements=ALLOWED_ELEMENTS,
    )
    decoder.index2z = {v: k for k, v in z2index.items()}
    model = model_mod.EncoderDecoderModel(
        atom_fea_dim=64,
        edge_fea_dim=128,
        depth=3,
        max_atom_num=MAX_ATOMS,
        extra_fea_dim=len(EXTRA_FEATURE_COLS),
        latent_dim=latent_dim,
        num_atom_types=NUM_ATOM_TYPES,
        decoder=decoder,
        prop_pred=model_mod.PropertyPredictor(latent_dim=latent_dim, cond_dim=len(COND_COLS)),
    )
    return model.to(device), latent_dim


def build_unconditional_model(model_mod, z2index, label2index, device):
    latent_dim = 64
    decoder = model_mod.StructureDecoder(
        latent_dim=latent_dim,
        max_atoms=MAX_ATOMS,
        num_atom_types=NUM_ATOM_TYPES,
        z2index=z2index,
        num_wp_labels=len(label2index),
        allowed_elements=ALLOWED_ELEMENTS,
    )
    decoder.index2z = {v: k for k, v in z2index.items()}
    model = EncoderDecoderUnconditional(
        model_mod=model_mod,
        atom_fea_dim=64,
        edge_fea_dim=128,
        depth=3,
        max_atom_num=MAX_ATOMS,
        num_atom_types=NUM_ATOM_TYPES,
        decoder=decoder,
        prop_pred=model_mod.PropertyPredictor(latent_dim=latent_dim, cond_dim=len(COND_COLS)),
    )
    return model.to(device), latent_dim


def map_lattice_params(lattice_norm_np):
    A_MIN, A_MAX = 3.0, 18.0
    ALPHA_MIN, ALPHA_MAX = 45.0, 135.0
    a_raw, b_raw, c_raw, alpha_raw, beta_raw, gamma_raw = lattice_norm_np
    a = A_MIN + (A_MAX - A_MIN) * a_raw
    b = A_MIN + (A_MAX - A_MIN) * b_raw
    c = A_MIN + (A_MAX - A_MIN) * c_raw
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * alpha_raw
    beta = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * beta_raw
    gamma = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * gamma_raw
    return a, b, c, alpha, beta, gamma


def is_pathological_lattice(lattice) -> bool:
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    vol = lattice.volume
    cond_number = np.linalg.cond(lattice.matrix)
    if min(a, b, c) < 1.0 or max(a, b, c) > 50.0:
        return True
    if any(angle < 10.0 or angle > 170.0 for angle in (alpha, beta, gamma)):
        return True
    if vol < 5.0 or not math.isfinite(vol):
        return True
    if cond_number > 1e6:
        return True
    return False


def structure_to_metadata_row(cif_path: str, formula: str, variant: str, pred_density=None, pred_melting=None):
    row = {
        "source_path": cif_path,
        "formula_pretty": formula,
        "variant": variant,
    }
    if pred_density is not None:
        row["pred_density"] = pred_density
    if pred_melting is not None:
        row["pred_melting_point_log"] = pred_melting
    return row


def decode_and_filter(model, latent_samples, output_dir: Path, variant: str, allowed_Z: set[int]):
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = output_dir / "generated_CIF"
    cif_dir.mkdir(parents=True, exist_ok=True)

    unique_structures = []
    matcher = StructureMatcher()
    metadata_rows = []

    with torch.no_grad():
        _, lattice_norm, coords, types_logits = model.decoder(latent_samples, mode="generate")
        prop_out = model.prop_pred(latent_samples) if model.prop_pred is not None else None

    for i in range(latent_samples.shape[0]):
        try:
            lattice = PmgLattice.from_parameters(*map_lattice_params(lattice_norm[i].detach().cpu().numpy()))
            if is_pathological_lattice(lattice):
                continue
        except Exception:
            continue

        coords_np = coords[i].detach().cpu().numpy()
        types_np = types_logits[i].argmax(dim=-1).detach().cpu().numpy()
        species, frac_coords = [], []
        for j, type_idx in enumerate(types_np.tolist()):
            if type_idx == 0:
                continue
            Z = model.decoder.index2z.get(int(type_idx))
            if Z is None or Z not in allowed_Z:
                continue
            species.append(Element.from_Z(int(Z)))
            frac_coords.append(coords_np[j].tolist())

        if len(set(species)) <= 1 or len(species) > 20 or len(set(species)) > 6:
            continue

        try:
            struct = Structure(lattice, species, frac_coords)
            dist = np.array(struct.distance_matrix, dtype=float)
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.1:
                continue
            analyzer = SpacegroupAnalyzer(struct, symprec=1e-2)
            std_struct = analyzer.get_conventional_standard_structure()
            if not std_struct.is_ordered:
                continue
            duplicate = any(matcher.fit(std_struct, s) for s in unique_structures)
            if duplicate:
                continue
            unique_structures.append(std_struct)

            formula = struct.composition.reduced_formula
            cif_path = cif_dir / f"generated_{i+1}_{formula}.cif"
            CifWriter(struct).write_file(str(cif_path))

            pred_density = None
            pred_melting = None
            if prop_out is not None:
                pred = prop_out[i].detach().cpu().numpy().reshape(-1)
                if pred.shape[0] >= 1:
                    pred_melting = float(pred[0])
                if pred.shape[0] >= 2:
                    pred_density = float(pred[1])

            metadata_rows.append(structure_to_metadata_row(str(cif_path), formula, variant, pred_density, pred_melting))
        except Exception:
            continue

    pd.DataFrame(metadata_rows).to_csv(output_dir / "generated_metadata.csv", index=False)
    print(f"[{variant}] Saved {len(metadata_rows)} metadata rows to {output_dir / 'generated_metadata.csv'}")


def run_full_guided_subprocess(module_path: str, weights: str, data_csv: str, output_dir: Path,
                               num_samples: int, cond_melting: float, cond_density: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", module_path,
        "--mode", "generate",
        "--load_endecoder", weights,
        "--generate_data_csv", data_csv,
        "--num_samples", str(num_samples),
        "--cond_melting", str(cond_melting),
        "--cond_density", str(cond_density),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    subprocess.run(cmd, check=True, env=env)
    # Non-destructive copy of the original output folder if present.
    original_cif_dir = Path("generated_CIF")
    if original_cif_dir.exists():
        target = output_dir / "generated_CIF"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(original_cif_dir, target)



def train_unconditional(model_mod, data_csv: str, save_path: str, epochs: int, batch_size: int, device):
    df = pd.read_csv(data_csv)
    df = filter_dataframe(df, model_mod, MAX_ATOMS, ALLOWED_ELEMENTS)
    z2index = build_z2index(df)
    df_train, _ = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    dataset = model_mod.StructureDataset(
        df_train,
        cond_cols=COND_COLS,
        extra_feature_cols=EXTRA_FEATURE_COLS,
        max_atoms=MAX_ATOMS,
        z2index=z2index,
    )
    label2index = build_label2index(dataset)
    if torch.cuda.is_available():
        dataset.device = torch.device("cuda")

    model, _ = build_unconditional_model(model_mod, z2index, label2index, device)
    loss_list = model_mod.train_encoder_decoder(
        model=model,
        dataset=dataset,
        label2index=label2index,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        device=device,
        lattice_w=10.0,
        coord_w=5.0,
        type_w=10.0,
        prop_w=0.0,
        patience=20,
        num_atom_types=NUM_ATOM_TYPES,
        save_path=save_path,
    )
    return loss_list


def main():
    parser = argparse.ArgumentParser(description="Benchmark-only generation branches for Comment #7.")
    parser.add_argument("--model_module_path", type=str, default="modelA.py")
    parser.add_argument("--variant", choices=["full_guided", "concat_only", "unconditional"], required=True)
    parser.add_argument("--action", choices=["generate", "train_unconditional"], default="generate")
    parser.add_argument("--weights", type=str, required=False, help="Weights for the selected variant")
    parser.add_argument("--init_from_full_weights", type=str, default=None, help="Optional initialization for unconditional benchmark")
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cond_melting", type=float, default=float(np.log(1400.0)))
    parser.add_argument("--cond_density", type=float, default=8.0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model_mod = load_model_module(args.model_module_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    if args.action == "train_unconditional":
        save_path = args.weights if args.weights else str(output_dir / "unconditional_model.pt")
        output_dir.mkdir(parents=True, exist_ok=True)
        train_unconditional(model_mod, args.data_csv, save_path, args.epochs, args.batch_size, device)
        if args.init_from_full_weights:
            # Optional warm start helper, applied after model creation but before training in a custom workflow.
            print("Note: --init_from_full_weights is not automatically applied inside train_unconditional().")
            print("Use partial_load_matching_weights() in a custom script if you want warm-start initialization.")
        return

    if args.variant == "full_guided":
        if not args.weights:
            raise ValueError("--weights is required for full_guided generation")
        run_full_guided_subprocess(
            module_path=args.model_module_path,
            weights=args.weights,
            data_csv=args.data_csv,
            output_dir=output_dir,
            num_samples=args.num_samples,
            cond_melting=args.cond_melting,
            cond_density=args.cond_density,
        )
        return

    df = pd.read_csv(args.data_csv)
    df = filter_dataframe(df, model_mod, MAX_ATOMS, ALLOWED_ELEMENTS)
    z2index = build_z2index(df)
    dataset = model_mod.StructureDataset(
        df,
        cond_cols=COND_COLS,
        extra_feature_cols=EXTRA_FEATURE_COLS,
        max_atoms=MAX_ATOMS,
        z2index=z2index,
    )
    label2index = build_label2index(dataset)
    if torch.cuda.is_available():
        dataset.device = torch.device("cuda")

    allowed_Z = set(Element(sym).Z for sym in ALLOWED_ELEMENTS)
    
    if args.variant == "concat_only":
        if not args.weights:
            raise ValueError("--weights is required for concat_only generation")
        model, latent_dim = build_full_model(model_mod, z2index, label2index, device)
        state = torch.load(args.weights, map_location=device, weights_only=True)

        loaded, skipped = partial_load_matching_weights(model, state)
        print(f"[INFO] concat_only: loaded {len(loaded)} tensors, skipped {len(skipped)} tensors")
        if skipped:
            print("[INFO] skipped keys:")
            for k in skipped:
                print("   ", k)

        model.eval()
        latent_samples = torch.randn(args.num_samples, latent_dim, device=device)
        decode_and_filter(model, latent_samples, output_dir, args.variant, allowed_Z)
        return


    if args.variant == "unconditional":
        if not args.weights:
            raise ValueError("--weights is required for unconditional generation")
        model, latent_dim = build_unconditional_model(model_mod, z2index, label2index, device)
        state = torch.load(args.weights, map_location=device, weights_only=True)
        if args.init_from_full_weights:
            full_state = torch.load(args.init_from_full_weights, map_location=device, weights_only=True)
            loaded, skipped = partial_load_matching_weights(model, full_state)
            print(f"Warm-start from full weights loaded {len(loaded)} tensors, skipped {len(skipped)}")
        loaded, skipped = partial_load_matching_weights(model, state)
        print(f"Loaded unconditional weights: {len(loaded)} tensors, skipped {len(skipped)}")
        model.eval()
        latent_samples = torch.randn(args.num_samples, latent_dim, device=device)
        decode_and_filter(model, latent_samples, output_dir, args.variant, allowed_Z)
        return


if __name__ == "__main__":
    main()
