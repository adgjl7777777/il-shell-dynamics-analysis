
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

CODE_ROOT = Path("/nas_2/transcendence/il_paper/code")
CLASSIFY_DIR = CODE_ROOT / "classification"
RESULTS_DIR = CODE_ROOT / "results"
OUT_DIR = Path("/nas_2/transcendence/revision/analysis/uncertainty_bootstrap/outputs")
NOTE_PATH = Path("/nas_2/transcendence/revision/notes/03_r2_other_science/beti298_sampling_risk.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]
TRAIL = "1.0"
TOTAL_STEPS = 100_000
N_BLOCKS = 5
BLOCK_SIZE = TOTAL_STEPS // N_BLOCKS
N_BOOT = 20_000
RNG = np.random.default_rng(20260530)


def read_csv(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_table_theta():
    out = {}
    for r in read_csv(RESULTS_DIR / "table2_theta.csv"):
        out[(r["anion"], int(r["T"]))] = (float(r["theta"]), float(r.get("theta_se", 0) or 0))
    return out


def load_diffusion():
    out = {}
    for r in read_csv(RESULTS_DIR / "table5_diffusion.csv"):
        out[(r["anion"], int(r["T"]))] = {k: float(r[k]) for k in ["D_soft_A2ps", "D_soft_se", "D_hard_A2ps", "D_hard_se", "D_total_A2ps", "D_total_se"]}
    return out


def read_segments(path: Path):
    segs = []
    if not path.exists():
        return segs
    with path.open() as fh:
        for line in fh:
            vals = [int(v) for v in line.split() if v.isdigit()]
            if len(vals) >= 2:
                segs.append((vals[0], vals[-1]))
    return segs


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def block_theta_units(anion: str, T: int):
    rows = []
    for ion in range(5):
        soft = read_segments(CLASSIFY_DIR / "result" / anion / "soft" / str(T) / f"{TRAIL}_{ion}.txt")
        hard = read_segments(CLASSIFY_DIR / "result" / anion / "hard" / str(T) / f"{TRAIL}_{ion}.txt")
        for b in range(N_BLOCKS):
            b0, b1 = b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE
            soft_d = sum(overlap(s, e, b0, b1) for s, e in soft)
            hard_d = sum(overlap(s, e, b0, b1) for s, e in hard)
            total = soft_d + hard_d
            theta = soft_d / total if total > 0 else math.nan
            rows.append({"anion": anion, "T": T, "ion": ion, "block": b, "theta": theta, "soft_duration": soft_d, "hard_duration": hard_d})
    return rows


def bootstrap_mean_ci(values, n_boot=N_BOOT):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(means.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def propagated_dsoft_error(theta, theta_se, d):
    D_hard = d["D_hard_A2ps"]
    D_total = d["D_total_A2ps"]
    D_hard_se = d["D_hard_se"]
    D_total_se = d["D_total_se"]
    D_soft = (D_total - (1 - theta) * D_hard) / theta
    err = math.sqrt(
        (D_total_se / theta) ** 2
        + ((1 - theta) / theta * D_hard_se) ** 2
        + ((D_hard - D_soft) / theta * theta_se) ** 2
    )
    return D_soft, err


def main():
    table_theta = load_table_theta()
    diffusion = load_diffusion()

    block_rows = []
    summary_rows = []
    for anion in ANIONS:
        for T in TEMPS:
            units = block_theta_units(anion, T)
            block_rows.extend(units)
            vals = np.asarray([r["theta"] for r in units if np.isfinite(r["theta"])], dtype=float)
            block_means = []
            for b in range(N_BLOCKS):
                bvals = [r["theta"] for r in units if r["block"] == b and np.isfinite(r["theta"])]
                block_means.append(float(np.mean(bvals)))
            ion_means = []
            for ion in range(5):
                ivals = [r["theta"] for r in units if r["ion"] == ion and np.isfinite(r["theta"])]
                ion_means.append(float(np.mean(ivals)))

            theta, theta_se_ion = table_theta[(anion, T)]
            block_unit_se, boot_lo, boot_hi = bootstrap_mean_ci(vals)
            timeblock_se = float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))
            ion_recomputed_se = float(np.std(ion_means, ddof=1) / math.sqrt(len(ion_means)))
            conservative_theta_se = max(theta_se_ion, block_unit_se, timeblock_se, ion_recomputed_se)
            D_soft_cons, D_soft_cons_se = propagated_dsoft_error(theta, conservative_theta_se, diffusion[(anion, T)])

            summary_rows.append({
                "anion": anion,
                "T": T,
                "theta": theta,
                "theta_se_table_per_ion": theta_se_ion,
                "theta_se_ion_recomputed": ion_recomputed_se,
                "theta_se_block_unit_bootstrap": block_unit_se,
                "theta_boot_lo_95": boot_lo,
                "theta_boot_hi_95": boot_hi,
                "theta_se_timeblock_means": timeblock_se,
                "theta_se_conservative_max": conservative_theta_se,
                "theta_ion_min": min(ion_means),
                "theta_ion_max": max(ion_means),
                "theta_timeblock_min": min(block_means),
                "theta_timeblock_max": max(block_means),
                "D_soft_A2ps_original": diffusion[(anion, T)]["D_soft_A2ps"],
                "D_soft_se_original": diffusion[(anion, T)]["D_soft_se"],
                "D_soft_A2ps_conservative_theta": D_soft_cons,
                "D_soft_se_conservative_theta": D_soft_cons_se,
            })

    block_csv = OUT_DIR / "theta_block_units.csv"
    with block_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["anion", "T", "ion", "block", "theta", "soft_duration", "hard_duration"])
        w.writeheader(); w.writerows(block_rows)

    summary_csv = OUT_DIR / "theta_uncertainty_summary.csv"
    fields = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)

    beti298 = next(r for r in summary_rows if r["anion"] == "beti" and r["T"] == 298)
    worst_rel = sorted(summary_rows, key=lambda r: r["theta_se_conservative_max"] / r["theta"], reverse=True)[:4]

    lines = []
    lines.append("# R2-5 Sampling, BETI 298 K Risk, and Block Uncertainty")
    lines.append("")
    lines.append("Date: 2026-05-30")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Reviewer 2 noted that each system contains only five Li+ ions and that bridging anions reduce the independence of the five ion trajectories. This note adds a practical block/ion uncertainty check from the available 100 ns trajectories, without claiming that it replaces independent replicas.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Canonical threshold eta = 1.0 was used.")
    lines.append("- Each 100 ns trajectory was split into five 20 ns blocks for each Li+ ion, giving 25 ion-block theta estimates per system.")
    lines.append("- Bootstrap confidence intervals were computed over the 25 ion-block values as a practical internal resampling check.")
    lines.append("- A conservative theta uncertainty was defined as the maximum of the original per-ion SE, recomputed ion SE, ion-block bootstrap SE, and time-block SE. This is still not a substitute for independent replicas, but it is more visible and more cautious than relying only on the nominal five-ion SE.")
    lines.append("")
    lines.append("## Key Readout")
    lines.append("")
    lines.append(f"- BETI 298 K remains the most fragile condition: theta = {beti298["theta"]:.4f}, original per-ion SE = {beti298["theta_se_table_per_ion"]:.4f}, conservative block/ion SE = {beti298["theta_se_conservative_max"]:.4f}.")
    lines.append(f"- BETI 298 K ion-to-ion theta range: {beti298["theta_ion_min"]:.4f} to {beti298["theta_ion_max"]:.4f}; 20 ns block-mean range: {beti298["theta_timeblock_min"]:.4f} to {beti298["theta_timeblock_max"]:.4f}.")
    lines.append(f"- BETI 298 K D_soft uncertainty grows from {beti298["D_soft_se_original"]:.2e} to {beti298["D_soft_se_conservative_theta"]:.2e} A^2/ps when the conservative theta SE is propagated.")
    lines.append("- Highest relative theta uncertainties under the conservative estimate:")
    for r in worst_rel:
        lines.append(f"  - {r["anion"].upper()} {r["T"]} K: theta = {r["theta"]:.4f}, SE = {r["theta_se_conservative_max"]:.4f}, relative SE = {r["theta_se_conservative_max"]/r["theta"]:.2f}")
    lines.append("")
    lines.append("## Response Strategy")
    lines.append("")
    lines.append("Use this to explicitly acknowledge sampling limits while showing that we performed an additional internal uncertainty check. The safest response is: we added block/ion resampling for theta, retained bootstrap CIs for burstiness parameters, marked five-ion standard errors as lower bounds because of bridging correlations, and flagged BETI 298 K near the relevant results rather than only in the limitations section.")
    lines.append("")
    lines.append("## Manuscript/SI Actions")
    lines.append("")
    lines.append("1. Add a compact SI table with theta, original SE, block/ion conservative SE, and the ion/block ranges.")
    lines.append("2. Add one sentence near the parameter table: BETI 298 K has the largest finite-sample uncertainty and should be interpreted cautiously.")
    lines.append("3. In captions, call the five-ion SE a lower-bound uncertainty where bridging correlations matter.")
    lines.append("4. Do not claim independent replicas unless new simulations exist.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Block-unit CSV: `{block_csv}`")
    lines.append(f"- Summary CSV: `{summary_csv}`")
    lines.append(f"- Script: `{Path(__file__)}`")
    NOTE_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {block_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
