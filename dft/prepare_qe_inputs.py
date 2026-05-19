import argparse
from pathlib import Path

from pymatgen.io.cif import CifParser
from pymatgen.core import Element


ECUTWFC = 40.0
ECUTRHO = 320.0
DEGAUSS = 0.02
SMEARING = "mp"
CONV_THR = "0.0001"
MIXING_MODE = "local-TF"
MIXING_BETA = 0.3
PRESS_CONV_THR = 0.5
CELL_FACTOR = 2.0
KPOINTS = (6, 6, 6)


PSEUDO_MAP = {
    "Li": "Li.UPF",
    "Be": "Be.UPF",
    "B": "B.UPF",
    "C": "C.UPF",
    "N": "N.UPF",
    "Na": "Na.UPF",
    "Mg": "Mg.UPF",
    "Al": "Al.UPF",
    "Si": "Si.UPF",
    "K": "K.UPF",
    "Ca": "Ca.UPF",
    "Sc": "Sc.UPF",
    "Ti": "Ti.UPF",
    "V": "V.UPF",
    "Cr": "Cr.UPF",
    "Mn": "Mn.UPF",
    "Fe": "Fe.UPF",
    "Co": "Co.UPF",
    "Ni": "Ni.UPF",
    "Cu": "Cu.UPF",
    "Zn": "Zn.UPF",
    "Ga": "Ga.UPF",
    "Ge": "Ge.UPF",
    "Rb": "Rb.UPF",
    "Sr": "Sr.UPF",
    "Y": "Y.UPF",
    "Zr": "Zr.UPF",
    "Nb": "Nb.UPF",
    "Mo": "Mo.UPF",
    "Tc": "Tc.UPF",
    "Ru": "Ru.UPF",
    "Rh": "Rh.UPF",
    "Pd": "Pd.UPF",
    "Ag": "Ag.UPF",
    "Cd": "Cd.UPF",
    "In": "In.UPF",
    "Sn": "Sn.UPF",
    "Sb": "Sb.UPF",
    "Hf": "Hf.UPF",
    "Ta": "Ta.UPF",
    "W": "W.UPF",
    "Re": "Re.UPF",
    "Os": "Os.UPF",
}


def read_structure(cif_path: Path):
    parser = CifParser(str(cif_path))
    structs = parser.get_structures(primitive=False)

    if not structs:
        raise ValueError(f"No structure parsed from {cif_path}")

    return structs[0]


def qe_input_text(structure, prefix: str, pseudo_dir: str, qe_outdir: str):
    species_order = []
    seen = set()

    for site in structure:
        el = site.specie.symbol
        if el not in seen:
            seen.add(el)
            species_order.append(el)

    missing = [el for el in species_order if el not in PSEUDO_MAP]
    if missing:
        raise KeyError(f"Missing pseudo mapping for: {missing}")

    lines = []

    lines.append("&CONTROL")
    lines.append("   calculation      = 'vc-relax'")
    lines.append("   verbosity        = 'high'")
    lines.append("   tstress          = .true.")
    lines.append("   tprnfor          = .true.")
    lines.append(f"   outdir           = '{qe_outdir}'")
    lines.append(f"   prefix           = '{prefix}'")
    lines.append(f"   pseudo_dir       = '{pseudo_dir}'")
    lines.append("/")

    lines.append("&SYSTEM")
    lines.append(f"   ecutwfc          = {ECUTWFC}")
    lines.append(f"   ecutrho          = {ECUTRHO}")
    lines.append("   occupations      = 'smearing'")
    lines.append(f"   degauss          = {DEGAUSS}")
    lines.append(f"   smearing         = '{SMEARING}'")
    lines.append(f"   ntyp             = {len(species_order)}")
    lines.append(f"   nat              = {len(structure)}")
    lines.append("   ibrav            = 0")
    lines.append("/")

    lines.append("&ELECTRONS")
    lines.append(f"   conv_thr         = {CONV_THR}")
    lines.append(f"   mixing_mode      = '{MIXING_MODE}'")
    lines.append(f"   mixing_beta      = {MIXING_BETA}")
    lines.append("/")

    lines.append("&IONS")
    lines.append("   ion_dynamics     = 'bfgs'")
    lines.append("/")

    lines.append("&CELL")
    lines.append("   cell_dynamics    = 'bfgs'")
    lines.append(f"   cell_factor      = {CELL_FACTOR}")
    lines.append(f"   press_conv_thr   = {PRESS_CONV_THR}")
    lines.append("/")

    lines.append("&FCP")
    lines.append("/")
    lines.append("&RISM")
    lines.append("/")

    lines.append("ATOMIC_SPECIES")
    for el in species_order:
        lines.append(f"{el} {Element(el).atomic_mass:.6f} {PSEUDO_MAP[el]}")

    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"{KPOINTS[0]} {KPOINTS[1]} {KPOINTS[2]}  0 0 0")

    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for row in structure.lattice.matrix:
        lines.append(f"{row[0]:.14f} {row[1]:.14f} {row[2]:.14f}")

    lines.append("")
    lines.append("ATOMIC_POSITIONS angstrom")
    for site in structure:
        x, y, z = site.coords
        lines.append(f"{site.specie.symbol} {x:.10f} {y:.10f} {z:.10f}")

    lines.append("")

    return "\n".join(lines)


def process_dir(input_cif_dir: Path, output_dir: Path, pseudo_dir: str, qe_outdir: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(input_cif_dir.glob("*.cif"))

    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {input_cif_dir}")

    for cif in cif_files:
        try:
            structure = read_structure(cif)
            prefix = cif.stem

            text = qe_input_text(
                structure=structure,
                prefix=prefix,
                pseudo_dir=pseudo_dir,
                qe_outdir=qe_outdir,
            )

            outfile = output_dir / f"{prefix}.in"
            outfile.write_text(text, encoding="utf-8")
            print(f"[OK] wrote {outfile}")

        except Exception as e:
            print(f"[FAIL] {cif}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIF files into Quantum ESPRESSO vc-relax input files."
    )

    parser.add_argument(
        "--input_cif_dir",
        required=True,
        help="Directory containing input CIF files.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where Quantum ESPRESSO input files will be saved.",
    )

    parser.add_argument(
        "--pseudo_dir",
        required=True,
        help="Directory containing pseudopotential files used by Quantum ESPRESSO.",
    )

    parser.add_argument(
        "--qe_outdir",
        required=True,
        help="Scratch/output directory used by Quantum ESPRESSO during calculation.",
    )

    args = parser.parse_args()

    process_dir(
        input_cif_dir=Path(args.input_cif_dir),
        output_dir=Path(args.output_dir),
        pseudo_dir=args.pseudo_dir,
        qe_outdir=args.qe_outdir,
    )


if __name__ == "__main__":
    main()
