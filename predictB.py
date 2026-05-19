import argparse
import json
import os

import torch
import pandas as pd
import numpy as np
from pymatgen.core import Structure

from modelB import CGCNN, build_cgcnn_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict melting point and density for generated materials "
            "using fine-tuned Model B weights."
        )
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file containing generated structures.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file for predicted properties.",
    )

    parser.add_argument(
        "--mp_model",
        type=str,
        default="modelB-weights/best_modelB_mp_finetuned.pth",
        help="Fine-tuned Model B weight file for melting-point prediction.",
    )

    parser.add_argument(
        "--rho_model",
        type=str,
        default="modelB-weights/best_modelB_rho_finetuned.pth",
        help="Fine-tuned Model B weight file for density prediction.",
    )

    return parser.parse_args()


def load_state_dict_safely(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # For compatibility with older PyTorch versions that do not support weights_only.
        return torch.load(model_path, map_location=device)


def predict_values(df_new, best_model_path, device):
    model = CGCNN(use_extra_fea=False)
    model.load_state_dict(load_state_dict_safely(best_model_path, device))
    model.to(device)
    model.eval()

    def predict_single_structure(structure):
        node_fea, edge_index, edge_fea = build_cgcnn_graph(structure)

        node_fea = torch.LongTensor(node_fea).to(device).unsqueeze(0)
        edge_index = torch.LongTensor(edge_index).to(device)
        edge_fea = torch.FloatTensor(edge_fea).to(device).unsqueeze(0)
        crystal_atom_idx = torch.zeros(
            node_fea.shape[1], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            pred = model(node_fea[0], edge_index, edge_fea[0], crystal_atom_idx)

        return pred.item()

    predictions = []

    for i, row in df_new.iterrows():
        try:
            pred = predict_single_structure(row["structure"])
        except Exception as e:
            structure = row["structure"]
            print(structure)
            print("Number of atoms:", len(structure))
            print("Neighbors at r=6.0:", structure.get_neighbor_list(r=6.0)[0].shape[0])
            print(f"Error at row {i} (formula_pretty: {row['formula_pretty']}): {e}")
            pred = np.nan

        predictions.append(pred)

    return predictions


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_new = pd.read_csv(args.input_csv)

    if "structure" not in df_new.columns:
        raise ValueError("Input CSV must contain a 'structure' column.")

    if "formula_pretty" not in df_new.columns:
        raise ValueError("Input CSV must contain a 'formula_pretty' column.")

    df_new["structure"] = df_new["structure"].apply(
        lambda x: Structure.from_dict(json.loads(x))
    )

    df_pred = pd.DataFrame()
    df_pred["formula_pretty"] = df_new["formula_pretty"]

    model_paths = {
        "melting_point_log": args.mp_model,
        "density": args.rho_model,
    }

    for target_col, model_path in model_paths.items():
        print(f"Predicting {target_col} using {model_path}")

        predictions = predict_values(df_new, model_path, device)

        if target_col == "melting_point_log":
            predictions = np.exp(predictions)
            df_pred["melting_point"] = predictions
        else:
            df_pred[target_col] = predictions

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_pred.to_csv(args.output_csv, index=False)
    print(f"Prediction finished. Results are saved to {args.output_csv}")


if __name__ == "__main__":
    main()