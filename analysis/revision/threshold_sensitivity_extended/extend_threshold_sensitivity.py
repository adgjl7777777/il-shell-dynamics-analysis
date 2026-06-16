
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

CODE_ROOT = Path("/nas_2/transcendence/il_paper/code")
CLASSIFY_DIR = CODE_ROOT / "classification"
RESULTS_DIR = CODE_ROOT / "results"
OUT_DIR = Path("/nas_2/transcendence/revision/analysis/threshold_sensitivity_extended/outputs")
NOTE_PATH = Path("/nas_2/transcendence/revision/notes/03_r2_other_science/threshold_sensitivity_extension.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]
CUTOFF = 50
KEY_TRAILS = [0.5, 1.0, 2.0]


def read_csv(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def key_float(x: float) -> str:
    return f"{x:.1f}"


def load_sensitivity_theta():
    out = {}
    for r in read_csv(RESULTS_DIR / "sensitivity_theta.csv"):
        out[(r["anion"], int(r["T"]), float(r["trail"]))] = {
            "theta": float(r["theta"]),
            "theta_se": float(r["theta_se"]),
        }
    return out


def load_sensitivity_alpha():
    out = {}
    for r in read_csv(RESULTS_DIR / "sensitivity_alpha.csv"):
        def f(v):
            return float(v) if v not in ("", None) else math.nan
        out[(r["anion"], int(r["T"]), float(r["trail"]))] = {
            "alpha_soft": f(r["alpha_soft"]),
            "alpha_soft_se": f(r["alpha_soft_se"]),
            "alpha_hard": f(r["alpha_hard"]),
            "alpha_hard_se": f(r["alpha_hard_se"]),
        }
    return out


def load_diffusion():
    out = {}
    for r in read_csv(RESULTS_DIR / "table5_diffusion.csv"):
        out[(r["anion"], int(r["T"]))] = {
            "D_hard": float(r["D_hard_A2ps"]),
            "D_total": float(r["D_total_A2ps"]),
        }
    return out


def available_trails(anion: str, T: int):
    d = CLASSIFY_DIR / "result" / anion / "soft" / str(T)
    trails = []
    if d.exists():
        for path in d.glob("*_0.txt"):
            try:
                trails.append(float(path.name.replace("_0.txt", "")))
            except ValueError:
                pass
    return sorted(set(trails))


def read_segments(path: Path):
    segs = []
    if not path.exists():
        return segs
    with path.open() as fh:
        for line in fh:
            vals = [int(v) for v in line.split() if v.isdigit()]
            if len(vals) >= 2:
                segs.append(vals)
    return segs


def burstiness_A(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    n = len(x)
    if n < 3:
        return math.nan
    mean = float(np.mean(x))
    if mean <= 0:
        return math.nan
    r = float(np.std(x) / mean)
    num = math.sqrt(n + 1) * r - math.sqrt(n - 1)
    den = (math.sqrt(n + 1) - 2) * r + math.sqrt(n - 1)
    return num / den if den != 0 else math.nan


def collect_state_metrics(anion: str, T: int, trail: float):
    metrics = {}
    for state in ["soft", "hard"]:
        shell_intervals = []
        durations = []
        n_segments = 0
        for ion in range(5):
            path = CLASSIFY_DIR / "result" / anion / state / str(T) / f"{key_float(trail)}_{ion}.txt"
            segs = read_segments(path)
            n_segments += len(segs)
            for seg in segs:
                for j in range(len(seg) - 1):
                    if seg[j] != 0 and seg[j + 1] != seg[j]:
                        shell_intervals.append(seg[j + 1] - seg[j])
                d = seg[-1] - seg[0] - CUTOFF
                if seg[0] != 0 and d > 0:
                    durations.append(d)
        metrics[f"A_shell_{state}"] = burstiness_A(shell_intervals)
        metrics[f"N_shell_{state}"] = len(shell_intervals)
        metrics[f"A_duration_{state}"] = burstiness_A(durations)
        metrics[f"N_duration_{state}"] = len(durations)
        metrics[f"N_segments_{state}"] = n_segments
    return metrics


def fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}g}"


def main():
    theta = load_sensitivity_theta()
    alpha = load_sensitivity_alpha()
    diff = load_diffusion()
    rows = []
    for anion in ANIONS:
        for T in TEMPS:
            for trail in available_trails(anion, T):
                key = (anion, T, trail)
                if key not in theta:
                    continue
                row = {"anion": anion, "T": T, "trail": trail}
                row.update(theta[key])
                row.update(alpha.get(key, {}))
                row.update(collect_state_metrics(anion, T, trail))
                d = diff[(anion, T)]
                th = row["theta"]
                D_hard = d["D_hard"]
                D_total = d["D_total"]
                row["D_hard_canonical_A2ps"] = D_hard
                row["D_total_A2ps"] = D_total
                row["D_soft_theta_only_A2ps"] = (D_total - (1 - th) * D_hard) / th if th > 0 else math.nan
                row["D_soft_over_D_hard_theta_only"] = row["D_soft_theta_only_A2ps"] / D_hard if D_hard > 0 else math.nan
                rows.append(row)

    fields = [
        "anion", "T", "trail", "theta", "theta_se",
        "alpha_soft", "alpha_soft_se", "alpha_hard", "alpha_hard_se",
        "A_shell_soft", "A_shell_hard", "N_shell_soft", "N_shell_hard",
        "A_duration_soft", "A_duration_hard", "N_duration_soft", "N_duration_hard",
        "N_segments_soft", "N_segments_hard",
        "D_total_A2ps", "D_hard_canonical_A2ps", "D_soft_theta_only_A2ps", "D_soft_over_D_hard_theta_only",
    ]
    full_csv = OUT_DIR / "threshold_sensitivity_extended.csv"
    with full_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    key_rows = [r for r in rows if any(abs(r["trail"] - kt) < 1e-9 for kt in KEY_TRAILS)]
    key_csv = OUT_DIR / "threshold_sensitivity_key_trails.csv"
    with key_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in key_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    def get(anion, T, trail):
        for r in rows:
            if r["anion"] == anion and r["T"] == T and abs(r["trail"] - trail) < 1e-9:
                return r
        return None

    canonical = [get(a, T, 1.0) for a in ANIONS for T in TEMPS]
    keyset = [get(a, T, tr) for a in ANIONS for T in TEMPS for tr in KEY_TRAILS]
    canonical = [r for r in canonical if r]
    keyset = [r for r in keyset if r]

    soft_gt_hard_shell = sum(1 for r in canonical if r["A_shell_soft"] > r["A_shell_hard"])
    soft_gt_hard_duration = sum(1 for r in canonical if r["A_duration_soft"] > r["A_duration_hard"])
    mobility_gt = sum(1 for r in canonical if r["D_soft_over_D_hard_theta_only"] > 1)
    mobility_gt_key = sum(1 for r in keyset if r["D_soft_over_D_hard_theta_only"] > 1)

    theta_ratios = []
    for a in ANIONS:
        for T in TEMPS:
            r05, r1, r2 = get(a, T, 0.5), get(a, T, 1.0), get(a, T, 2.0)
            if r05 and r1 and r2:
                theta_ratios.append((r2["theta"] / r05["theta"], r05["theta"], r1["theta"], r2["theta"], a, T))
    min_ratio = min(x[0] for x in theta_ratios)
    max_ratio = max(x[0] for x in theta_ratios)

    lines = []
    lines.append("# R2-2 Threshold-Sensitivity Extension")
    lines.append("")
    lines.append("Date: 2026-05-30")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Reviewer 2 asked that threshold-multiplier dependence be shown more explicitly, including the soft-state fraction, transition rates, burstiness parameters, and mobility estimates. The original SI already contained theta and alpha(h_soft/h_hard). This extension adds burstiness descriptors and a first-order mobility sensitivity using the existing threshold-resolved state classifications.")
    lines.append("")
    lines.append("## Data and Method")
    lines.append("")
    lines.append("- Inputs: existing `classification/result/{anion}/{soft|hard}/{T}/{eta}_{ion}.txt` files and existing `results/sensitivity_theta.csv`, `results/sensitivity_alpha.csv`, and `results/table5_diffusion.csv`.")
    lines.append("- Thresholds: all available eta values were scanned; the response-facing compact view should focus on eta = 0.5, 1.0, and 2.0.")
    lines.append("- New burstiness metrics: finite-size-corrected A_N was recomputed from threshold-resolved shell-change inter-event intervals and state-residence durations.")
    lines.append("- Mobility sensitivity: D_soft(eta) was recomputed from the population-weighted relation using theta(eta) while holding D_total and D_hard at their canonical fitted values. This is a first-order sensitivity of the inferred soft mobility to the classification threshold, not a full threshold-resolved MSD refit.")
    lines.append("")
    lines.append("## Key Readout")
    lines.append("")
    lines.append(f"- Canonical eta = 1.0: A_shell_soft > A_shell_hard in {soft_gt_hard_shell}/12 systems.")
    lines.append(f"- Canonical eta = 1.0: A_duration_soft > A_duration_hard in {soft_gt_hard_duration}/12 systems. This one is not the primary burstiness claim; duration statistics are expected to be comparatively closer to single-rate behavior.")
    lines.append(f"- Canonical eta = 1.0: the inferred D_soft/D_hard ratio is > 1 in {mobility_gt}/12 systems.")
    lines.append(f"- Across eta = 0.5, 1.0, and 2.0, the inferred D_soft/D_hard ratio is > 1 in {mobility_gt_key}/{len(keyset)} system-threshold combinations.")
    lines.append(f"- theta increases monotonically with eta by construction; theta(2.0)/theta(0.5) ranges from {min_ratio:.2f} to {max_ratio:.2f} across the 12 systems. This means absolute theta is threshold-dependent, so the manuscript should emphasize qualitative trends rather than exact universality of theta.")
    lines.append("")
    lines.append("## Response Strategy")
    lines.append("")
    lines.append("Use this as a controlled algorithm-dependence answer, not as proof that eta is irrelevant. The safe wording is that the qualitative soft/hard separation and enhanced soft-state mobility are retained over moderate threshold changes, while absolute theta and fitted rates vary with eta as expected for an operational classifier.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Full CSV: `{full_csv}`")
    lines.append(f"- Compact key-threshold CSV: `{key_csv}`")
    lines.append(f"- Script: `{Path(__file__)}`")
    lines.append("")
    lines.append("## Manuscript/SI Actions")
    lines.append("")
    lines.append("1. Replace the current three-panel SI sensitivity figure or add a companion table/panel containing A_N and D_soft/D_hard at eta = 0.5, 1.0, and 2.0.")
    lines.append("2. Add one main-text sentence: the classification threshold changes the absolute soft-state population but preserves the qualitative separation between bursty exchange-rich soft intervals and slower hard intervals over moderate eta changes.")
    lines.append("3. State the mobility sensitivity caveat: threshold-resolved theta was varied directly, whereas full threshold-resolved hard-state MSD refits were not generated for every eta.")
    NOTE_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {full_csv}")
    print(f"Wrote {key_csv}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
