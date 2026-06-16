
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

CODE_ROOT = Path("/nas_2/transcendence/il_paper/code")
RESULTS_DIR = CODE_ROOT / "results"
RDF_MIN = CODE_ROOT / "auxiliary/rdf/min.txt"
OUT_DIR = Path("/nas_2/transcendence/revision/analysis/beti_validation/outputs")
NOTE_PATH = Path("/nas_2/transcendence/revision/notes/03_r2_other_science/beti_forcefield_validation.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]

BOX_PATHS = {
    "fsi": {T: Path(f"/nas_2/transcendence/_delete/cowork/wmi-md/NPT/Temp/{T}/coords.out") for T in TEMPS},
    "tfsi": {T: Path(f"/nas_2/transcendence/_delete/cowork/tfsi/prd/{T}K/coords.out") for T in TEMPS},
    "beti": {T: Path(f"/nas_2/transcendence/_delete/cowork/wmi-md/NPT/Temp_BETI/{T}/coords.out") for T in TEMPS},
}

MASS = {
    "Li": 6.94,
    "pyr14": 142.263,
    "fsi": 180.132,
    "tfsi": 280.146,
    "beti": 380.162,
}


def read_csv(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def parse_rdf_minima():
    seen = set()
    rows = []
    with RDF_MIN.open() as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            anion, T, rmin, gmin = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
            key = (anion, T)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"anion": anion, "T": T, "rdf_first_min_A": rmin, "g_at_min": gmin})
    return rows


def parse_box_length(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if "Box (Angstroms)" in line and i + 1 < len(lines):
            vals = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", lines[i + 1])]
            if vals:
                return vals[0]
    raise ValueError(f"Box length not found in {path}")


def density_g_cm3(anion: str, L_A: float):
    mass_amu = 95 * MASS["pyr14"] + 5 * MASS["Li"] + 100 * MASS[anion]
    volume_A3 = L_A ** 3
    return mass_amu * 1.66053906660 / volume_A3


def load_diffusion_theta():
    diff = {}
    for r in read_csv(RESULTS_DIR / "table5_diffusion.csv"):
        diff[(r["anion"], int(r["T"]))] = {
            "theta": float(r["theta"]),
            "D_total_A2ps": float(r["D_total_A2ps"]),
            "D_hard_A2ps": float(r["D_hard_A2ps"]),
            "D_soft_A2ps": float(r["D_soft_A2ps"]),
        }
    return diff


def main():
    rdf_rows = parse_rdf_minima()
    diff = load_diffusion_theta()
    box_rows = []
    for anion in ANIONS:
        for T in TEMPS:
            L = parse_box_length(BOX_PATHS[anion][T])
            box_rows.append({
                "anion": anion,
                "T": T,
                "box_length_A_final_NPT_snapshot": L,
                "volume_A3": L ** 3,
                "estimated_density_g_cm3_final_NPT_snapshot": density_g_cm3(anion, L),
                "source": str(BOX_PATHS[anion][T]),
            })

    merged = []
    for anion in ANIONS:
        for T in TEMPS:
            r = next(x for x in rdf_rows if x["anion"] == anion and x["T"] == T)
            b = next(x for x in box_rows if x["anion"] == anion and x["T"] == T)
            d = diff[(anion, T)]
            merged.append({**r, **{k: v for k, v in b.items() if k not in ["anion", "T"]}, **d})

    def write(path, rows):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    rdf_csv = OUT_DIR / "rdf_first_minima.csv"
    box_csv = OUT_DIR / "npt_box_density_estimates.csv"
    merged_csv = OUT_DIR / "beti_internal_validation_summary.csv"
    write(rdf_csv, rdf_rows)
    write(box_csv, box_rows)
    write(merged_csv, merged)

    beti_rmins = [r["rdf_first_min_A"] for r in rdf_rows if r["anion"] == "beti"]
    fsi_rmins = [r["rdf_first_min_A"] for r in rdf_rows if r["anion"] == "fsi"]
    tfsi_rmins = [r["rdf_first_min_A"] for r in rdf_rows if r["anion"] == "tfsi"]
    density_298 = {r["anion"]: r["estimated_density_g_cm3_final_NPT_snapshot"] for r in box_rows if r["T"] == 298}
    D298 = {a: diff[(a, 298)]["D_total_A2ps"] for a in ANIONS}
    L298 = {r["anion"]: r["box_length_A_final_NPT_snapshot"] for r in box_rows if r["T"] == 298}

    lines = []
    lines.append("# R2-6 BETI Force-Field Validation Evidence")
    lines.append("")
    lines.append("Date: 2026-05-30")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Reviewer 2 requested stronger validation of the BETI- force-field treatment because BETI parameters were assigned within APPLE&P/SystemGenerator by analogy with TFSI-. This note collects internal consistency evidence already available locally and identifies what remains to be added from external literature.")
    lines.append("")
    lines.append("## Internal Evidence Collected")
    lines.append("")
    lines.append(f"- Li-anion RDF first minima: FSI {min(fsi_rmins):.3f}-{max(fsi_rmins):.3f} A, TFSI {min(tfsi_rmins):.3f}-{max(tfsi_rmins):.3f} A, BETI {min(beti_rmins):.3f}-{max(beti_rmins):.3f} A.")
    lines.append("- BETI has a slightly outward-shifted first minimum relative to FSI/TFSI, which is physically consistent with its longer perfluoroethyl substituents. This is useful validation, but it also means the current manuscript sentence saying all first minima fall in 5.4-5.6 A should be corrected.")
    lines.append(f"- Final NPT snapshot box lengths at 298 K increase with anion size: FSI {L298["fsi"]:.3f} A, TFSI {L298["tfsi"]:.3f} A, BETI {L298["beti"]:.3f} A.")
    lines.append(f"- Approximate densities from those final NPT snapshot boxes at 298 K are FSI {density_298["fsi"]:.3f}, TFSI {density_298["tfsi"]:.3f}, BETI {density_298["beti"]:.3f} g/cm3. These are snapshot-derived internal checks, not ensemble-averaged experimental validation.")
    lines.append(f"- Total Li diffusion at 298 K: FSI {D298["fsi"]:.3e}, TFSI {D298["tfsi"]:.3e}, BETI {D298["beti"]:.3e} A^2/ps. BETI is the slowest system, consistent with the bulkier anion producing slower transport in the simulations.")
    lines.append("")
    lines.append("## Safe Interpretation")
    lines.append("")
    lines.append("The strongest defensible statement is not that BETI is fully validated experimentally. Instead: BETI was parameterized with the same APPLE&P/SystemGenerator protocol as FSI/TFSI; the resulting RDF, box-size/density, coordination, and diffusion trends are internally consistent with the expected larger-anion behavior; exact experimental validation for this same Li+/Pyr14+/BETI composition should be stated as limited if no direct literature data are found.")
    lines.append("")
    lines.append("## Required Manuscript Changes")
    lines.append("")
    lines.append("1. Correct the RDF first-minimum sentence. Safer: first minima remain near the first-shell boundary, with BETI shifted slightly outward (about 5.7-5.9 A); a common 5.5 A cutoff was retained for consistent cross-system classification, and sensitivity/structural checks are provided.")
    lines.append("2. Add a short BETI validation paragraph in Simulation Details or SI: APPLE&P/SystemGenerator consistency, RDF first-shell continuity, box/density trend, and diffusion/coordination consistency.")
    lines.append("3. Add external literature if available: density, diffusion, viscosity, RDF, or structural data for Pyr14BETI, LiBETI/Pyr14BETI, or closely related sulfonylimide ILs.")
    lines.append("4. Avoid overclaiming: say internal consistency supports the BETI treatment, while exact composition-level validation remains a limitation.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- RDF minima CSV: `{rdf_csv}`")
    lines.append(f"- Box/density CSV: `{box_csv}`")
    lines.append(f"- Merged validation CSV: `{merged_csv}`")
    lines.append(f"- Script: `{Path(__file__)}`")
    NOTE_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {rdf_csv}")
    print(f"Wrote {box_csv}")
    print(f"Wrote {merged_csv}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
