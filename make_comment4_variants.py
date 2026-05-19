import argparse
from pathlib import Path
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter


def read_structure(cif_path: Path) -> Structure:
    return Structure.from_file(str(cif_path))


def perturb_coords_cartesian(structure: Structure, amplitude_ang: float, seed: int) -> Structure:
    rng = np.random.default_rng(seed)
    s = structure.copy()
    cart = np.array([site.coords for site in s.sites], dtype=float)
    disp = rng.uniform(-amplitude_ang, amplitude_ang, size=cart.shape)
    new_cart = cart + disp
    new_frac = s.lattice.get_fractional_coords(new_cart)
    new_frac = new_frac % 1.0

    new_s = Structure(
        lattice=s.lattice,
        species=[site.specie for site in s.sites],
        coords=new_frac,
        coords_are_cartesian=False,
    )
    return new_s


def perturb_lattice(structure: Structure, length_frac: float, angle_deg: float, seed: int) -> Structure:
    rng = np.random.default_rng(seed)
    s = structure.copy()

    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles

    da = 1.0 + rng.uniform(-length_frac, length_frac)
    db = 1.0 + rng.uniform(-length_frac, length_frac)
    dc = 1.0 + rng.uniform(-length_frac, length_frac)

    dalpha = rng.uniform(-angle_deg, angle_deg)
    dbeta = rng.uniform(-angle_deg, angle_deg)
    dgamma = rng.uniform(-angle_deg, angle_deg)

    new_a = a * da
    new_b = b * db
    new_c = c * dc
    new_alpha = alpha + dalpha
    new_beta = beta + dbeta
    new_gamma = gamma + dgamma

    # 简单保护，避免极端角度
    new_alpha = float(np.clip(new_alpha, 30.0, 150.0))
    new_beta = float(np.clip(new_beta, 30.0, 150.0))
    new_gamma = float(np.clip(new_gamma, 30.0, 150.0))

    new_lattice = Lattice.from_parameters(new_a, new_b, new_c, new_alpha, new_beta, new_gamma)

    new_s = Structure(
        lattice=new_lattice,
        species=[site.specie for site in s.sites],
        coords=[site.frac_coords for site in s.sites],
        coords_are_cartesian=False,
    )
    return new_s


def write_cif(structure: Structure, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(structure).write_file(str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing the 5 original CIFs")
    parser.add_argument("--output_dir", required=True, help="Directory to write variants")
    parser.add_argument("--base_seed", type=int, default=20260330)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    cif_files = sorted(input_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {input_dir}")

    for idx, cif_path in enumerate(cif_files):
        s = read_structure(cif_path)
        stem = cif_path.stem
        cand_dir = output_dir / stem
        cand_dir.mkdir(parents=True, exist_ok=True)

        # 1. original
        write_cif(s, cand_dir / "orig.cif")

        # 2-5. coordinate perturbations
        write_cif(
            perturb_coords_cartesian(s, amplitude_ang=0.05, seed=args.base_seed + idx * 100 + 1),
            cand_dir / "coord_005_a.cif"
        )
        write_cif(
            perturb_coords_cartesian(s, amplitude_ang=0.05, seed=args.base_seed + idx * 100 + 2),
            cand_dir / "coord_005_b.cif"
        )
        write_cif(
            perturb_coords_cartesian(s, amplitude_ang=0.10, seed=args.base_seed + idx * 100 + 3),
            cand_dir / "coord_010_a.cif"
        )
        write_cif(
            perturb_coords_cartesian(s, amplitude_ang=0.10, seed=args.base_seed + idx * 100 + 4),
            cand_dir / "coord_010_b.cif"
        )

        # 6-7. lattice perturbations
        write_cif(
            perturb_lattice(s, length_frac=0.01, angle_deg=1.0, seed=args.base_seed + idx * 100 + 5),
            cand_dir / "lat_1pct_1deg.cif"
        )
        write_cif(
            perturb_lattice(s, length_frac=0.03, angle_deg=3.0, seed=args.base_seed + idx * 100 + 6),
            cand_dir / "lat_3pct_3deg.cif"
        )

        print(f"[OK] wrote 7 variants for {stem} -> {cand_dir}")


if __name__ == "__main__":
    main()