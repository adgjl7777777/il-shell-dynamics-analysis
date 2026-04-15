"""
Sensitivity analysis for the classification threshold multiplier (A2-1).

The canonical classification uses TRAIL = 1.0 (Step 3 threshold = 1×Δt,
Step 4 merge criterion = 2×Δt). This script tests alternative multipliers
to confirm that the key conclusions are robust to this choice.

For each multiplier value it computes:
  - theta (soft-state fraction)
  - alpha(h_soft), alpha(h_hard) (exponential decay rates)

Pre-requisite: classification/result/{anion}/{soft|hard}/{T}/{TRAIL}_{i}.txt
must already exist for all tested TRAIL values.  The canonical pipeline
(Echecker.py) produces these files for TRAIL in 0.1, 0.2, …, 2.0, 2.5, …
The available multipliers are detected automatically from the result directory.

Output:
  results/sensitivity_theta.csv
  results/sensitivity_alpha.csv
  figures/sensitivity/ — one table-figure per anion (PNG)

Usage:
    cd path/to/il_paper/code
    python run_sensitivity.py
"""
import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt

CODE_ROOT    = os.path.dirname(os.path.abspath(__file__))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")
RESULTS_DIR  = os.path.join(CODE_ROOT, "results")
FIG_DIR      = os.path.join(CODE_ROOT, "..", "paper", "Images", "sensitivity")
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "classification"))

from config import ANIONS, TEMPERATURES
import dist

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CUTOFF = 50   # same as duration.py

# ── All systems ───────────────────────────────────────────────────────────────
TEST_SYSTEMS = [(a, T) for a in ANIONS for T in TEMPERATURES]

# Temperature → color (consistent with existing paper figures)
T_COLORS = {298: "#5555FF", 353: "#55FF55", 373: "#FFAA55", 423: "#FF5555"}
T_LABELS = {298: "298 K", 353: "353 K", 373: "373 K", 423: "423 K"}

# ── Detect available TRAIL values from the result directory ──────────────────
def get_available_trails(anion, T):
    d = os.path.join(CLASSIFY_DIR, "result", anion, "soft", str(T))
    trails = set()
    if os.path.isdir(d):
        for fname in os.listdir(d):
            if fname.endswith("_0.txt"):
                try:
                    trails.add(float(fname.replace("_0.txt", "")))
                except ValueError:
                    pass
    return sorted(trails)

# ── Duration extractor (mirrors duration.py logic) ───────────────────────────
def get_durations(anion, T, trail):
    T_str = str(T)
    realsoft, realhard = [], []
    for i in range(5):
        for state, container in [("soft", realsoft), ("hard", realhard)]:
            path = os.path.join(CLASSIFY_DIR, "result", anion, state, T_str,
                                f"{trail}_{i}.txt")
            if not os.path.exists(path):
                return None, None
            with open(path) as fh:
                segs = [[int(v) for v in line.strip().split() if v.isdigit()]
                        for line in fh]
            container.extend([s[-1] - s[0] - CUTOFF for s in segs
                               if s[0] != 0 and s[-1] - s[0] - CUTOFF > 0])
    return realsoft, realhard

# ── Theta from raw result files ───────────────────────────────────────────────
def get_theta(anion, T, trail):
    T_str = str(T)
    theta_per_atom = []
    for i in range(5):
        soft_i = hard_i = 0
        for state in ["soft", "hard"]:
            path = os.path.join(CLASSIFY_DIR, "result", anion, state, T_str,
                                f"{trail}_{i}.txt")
            if not os.path.exists(path):
                return None, None
            with open(path) as fh:
                for line in fh:
                    tokens = [v for v in line.strip().split() if v.isdigit()]
                    if len(tokens) >= 2:
                        dur = int(tokens[-1]) - int(tokens[0])
                        if state == "soft":
                            soft_i += dur
                        else:
                            hard_i += dur
        total_i = soft_i + hard_i
        if total_i > 0:
            theta_per_atom.append(soft_i / total_i)
    if not theta_per_atom:
        return None, None
    theta    = np.mean(theta_per_atom)
    theta_se = np.std(theta_per_atom, ddof=1) / np.sqrt(len(theta_per_atom))
    return theta, theta_se

# ── Main loop ─────────────────────────────────────────────────────────────────
theta_rows = []
alpha_rows = []

canonical = 1.0

for anion, T in TEST_SYSTEMS:
    trails = get_available_trails(anion, T)
    print(f"\n{anion} {T}K — {len(trails)} trail values: {trails}")

    for trail in trails:
        # theta
        theta, theta_se = get_theta(anion, T, trail)
        if theta is None:
            print(f"  trail={trail}: missing files, skipped")
            continue
        theta_rows.append({
            "anion": anion, "T": T, "trail": trail,
            "theta": round(theta, 6), "theta_se": round(theta_se, 6),
        })

        # alpha from h(n) fit
        realsoft, realhard = get_durations(anion, T, trail)
        a_soft = a_soft_se = a_hard = a_hard_se = float("nan")
        if realsoft and len(realsoft) > 5:
            _, _, a_soft, a_soft_se, _, _, ok = dist.h_calc(realsoft)
            if not ok:
                a_soft = a_soft_se = float("nan")
        if realhard and len(realhard) > 5:
            _, _, a_hard, a_hard_se, _, _, ok = dist.h_calc(realhard)
            if not ok:
                a_hard = a_hard_se = float("nan")

        alpha_rows.append({
            "anion": anion, "T": T, "trail": trail,
            "alpha_soft": round(a_soft, 8) if np.isfinite(a_soft) else "",
            "alpha_soft_se": round(a_soft_se, 8) if np.isfinite(a_soft_se) else "",
            "alpha_hard": round(a_hard, 8) if np.isfinite(a_hard) else "",
            "alpha_hard_se": round(a_hard_se, 8) if np.isfinite(a_hard_se) else "",
        })
        print(f"  trail={trail:5.1f}  theta={theta:.4f}±{theta_se:.4f}"
              f"  α_soft={a_soft:.3e}  α_hard={a_hard:.3e}")

# ── Write CSVs ────────────────────────────────────────────────────────────────
theta_csv = os.path.join(RESULTS_DIR, "sensitivity_theta.csv")
with open(theta_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["anion","T","trail","theta","theta_se"])
    w.writeheader(); w.writerows(theta_rows)
print(f"\nWritten: {theta_csv}")

alpha_csv = os.path.join(RESULTS_DIR, "sensitivity_alpha.csv")
with open(alpha_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["anion","T","trail",
                                        "alpha_soft","alpha_soft_se",
                                        "alpha_hard","alpha_hard_se"])
    w.writeheader(); w.writerows(alpha_rows)
print(f"Written: {alpha_csv}")

# ── Plots: theta vs trail for each system ─────────────────────────────────────
import pandas as pd
df_theta = pd.DataFrame(theta_rows)
df_alpha = pd.DataFrame(alpha_rows)

# convert alpha columns to numeric
for col in ["alpha_soft","alpha_soft_se","alpha_hard","alpha_hard_se"]:
    df_alpha[col] = pd.to_numeric(df_alpha[col], errors="coerce")

import math

def nice_bound(val, direction):
    """Return nearest n×10^k just below (direction=-1) or above (direction=+1) val."""
    if val <= 0:
        return val
    exp  = math.floor(math.log10(val))
    mant = val / 10**exp          # 1 ≤ mant < 10
    if direction == -1:
        n = math.floor(mant)
        if n < 1: n = 1
    else:
        n = math.ceil(mant)
        if n > 9: n, exp = 1, exp + 1
    return n * 10**exp

def make_log_ticks(lo, hi):
    exp_lo = math.floor(math.log10(lo))
    exp_hi = math.ceil(math.log10(hi))
    return [10**e for e in range(exp_lo, exp_hi + 1)]

# ── Grand alpha range across ALL anions / temps / states ─────────────────────
_all_alpha = pd.to_numeric(
    pd.concat([df_alpha["alpha_soft"], df_alpha["alpha_hard"]]), errors="coerce"
).dropna()
_all_alpha = _all_alpha[_all_alpha > 0]
ALPHA_GRAND_MIN = nice_bound(_all_alpha.min(), -1)
ALPHA_GRAND_MAX = nice_bound(_all_alpha.max(), +1)
ALPHA_RANGE  = (ALPHA_GRAND_MIN, ALPHA_GRAND_MAX)
ALPHA_TICKS  = make_log_ticks(ALPHA_GRAND_MIN, ALPHA_GRAND_MAX)
print(f"Global alpha range: {ALPHA_GRAND_MIN:.1e} – {ALPHA_GRAND_MAX:.1e}")

for anion in ANIONS:
    sub_t = df_theta[df_theta["anion"]==anion]
    sub_a = df_alpha[df_alpha["anion"]==anion]

    PANEL_CFG = [
        ("theta",      r"$\theta$",
         (1e-2, 1e0),
         make_log_ticks(1e-2, 1e0)),
        ("alpha_soft", r"$\alpha(h_\mathrm{soft})\ [\mathrm{ps}^{-1}]$",
         ALPHA_RANGE, ALPHA_TICKS),
        ("alpha_hard", r"$\alpha(h_\mathrm{hard})\ [\mathrm{ps}^{-1}]$",
         ALPHA_RANGE, ALPHA_TICKS),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Threshold sensitivity — {anion.upper()}$^-$", fontsize=12)

    for ax, (col, ylabel, ylim, yticks) in zip(axes, PANEL_CFG):
        for T in TEMPERATURES:
            c = T_COLORS[T]
            lbl = T_LABELS[T]
            if col == "theta":
                s = sub_t[sub_t["T"]==T].sort_values("trail")
                if s.empty: continue
                ax.errorbar(s["trail"], s["theta"], yerr=s["theta_se"],
                            color=c, marker="o", ms=3.5, lw=1.2,
                            capsize=2, label=lbl)
            else:
                s = sub_a[sub_a["T"]==T].sort_values("trail")
                if s.empty: continue
                se_col = col + "_se"
                vals = s[col]
                errs = s[se_col]
                # mask NaN
                mask = vals.notna() & (vals > 0)
                ax.errorbar(s["trail"][mask], vals[mask], yerr=errs[mask],
                            color=c, marker="o", ms=3.5, lw=1.2,
                            capsize=2, label=lbl)

        ax.axvline(canonical, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterMathtext())
        ax.set_xlabel("threshold multiplier", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, framealpha=0.7)

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, f"sensitivity_{anion}.pdf")
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Figure: {out_fig}")

print("\nDone.")
