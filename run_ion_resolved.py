"""
Ion-resolved analysis for 5 individual Li+ ions (A1-1 SI).

For each (anion, T, ion_index) computes:
  - theta_i  : soft-state fraction for that ion alone
  - alpha_soft_i, alpha_hard_i : exponential decay rates (from h_calc)

Outputs:
  results/ion_resolved_theta.csv
  results/ion_resolved_alpha.csv
  paper/Images/sensitivity/ion_resolved_theta.pdf   — theta per ion (bar chart)
  paper/Images/sensitivity/ion_resolved_timeline_{anion}.pdf — state timeline

Usage:
    cd path/to/il_paper/code
    python run_ion_resolved.py
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

TRAIL  = "1.0"
CUTOFF = 50   # same as duration.py

T_COLORS = {298: "#5555FF", 353: "#55FF55", 373: "#FFAA55", 423: "#FF5555"}
T_LABELS = {298: "298 K", 353: "353 K", 373: "373 K", 423: "423 K"}


# ── Helper: parse a result file → list of (start, end) segments ──────────────
def read_segments(path):
    segs = []
    with open(path) as fh:
        for line in fh:
            tokens = [int(v) for v in line.strip().split() if v.isdigit()]
            if len(tokens) >= 2:
                segs.append((tokens[0], tokens[-1]))
    return segs


# ── Helper: durations for h(n) fitting (mirror run_sensitivity.py logic) ─────
def seg_durations(segs):
    return [end - start - CUTOFF for start, end in segs
            if start != 0 and end - start - CUTOFF > 0]


# ── Main: collect per-ion stats ───────────────────────────────────────────────
theta_rows = []
alpha_rows = []

for anion in ANIONS:
    for T in TEMPERATURES:
        T_str = str(T)
        for i in range(5):
            soft_path = os.path.join(CLASSIFY_DIR, "result", anion, "soft", T_str,
                                     f"{TRAIL}_{i}.txt")
            hard_path = os.path.join(CLASSIFY_DIR, "result", anion, "hard", T_str,
                                     f"{TRAIL}_{i}.txt")
            if not os.path.exists(soft_path) or not os.path.exists(hard_path):
                print(f"Missing: {anion} {T}K ion {i}")
                continue

            soft_segs = read_segments(soft_path)
            hard_segs = read_segments(hard_path)

            # theta_i
            soft_dur = sum(e - s for s, e in soft_segs)
            hard_dur = sum(e - s for s, e in hard_segs)
            total    = soft_dur + hard_dur
            theta_i  = soft_dur / total if total > 0 else float("nan")
            theta_rows.append({"anion": anion, "T": T, "ion": i,
                                "theta": round(theta_i, 6)})

            # alpha_i
            soft_d = seg_durations(soft_segs)
            hard_d = seg_durations(hard_segs)
            a_soft = a_soft_se = a_hard = a_hard_se = float("nan")
            if len(soft_d) > 5:
                _, _, a_soft, a_soft_se, _, _, ok = dist.h_calc(soft_d)
                if not ok: a_soft = a_soft_se = float("nan")
            if len(hard_d) > 5:
                _, _, a_hard, a_hard_se, _, _, ok = dist.h_calc(hard_d)
                if not ok: a_hard = a_hard_se = float("nan")
            alpha_rows.append({"anion": anion, "T": T, "ion": i,
                                "alpha_soft": round(a_soft, 8) if np.isfinite(a_soft) else "",
                                "alpha_soft_se": round(a_soft_se, 8) if np.isfinite(a_soft_se) else "",
                                "alpha_hard": round(a_hard, 8) if np.isfinite(a_hard) else "",
                                "alpha_hard_se": round(a_hard_se, 8) if np.isfinite(a_hard_se) else ""})

        # Print per-system summary
        sub = [r for r in theta_rows if r["anion"] == anion and r["T"] == T]
        thetas = [r["theta"] for r in sub]
        if thetas:
            print(f"{anion:4s} {T}K  theta/ion: "
                  + "  ".join(f"{t:.4f}" for t in thetas)
                  + f"  mean={np.mean(thetas):.4f} std={np.std(thetas,ddof=1):.4f}")

# ── Write CSVs ────────────────────────────────────────────────────────────────
theta_csv = os.path.join(RESULTS_DIR, "ion_resolved_theta.csv")
with open(theta_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["anion","T","ion","theta"])
    w.writeheader(); w.writerows(theta_rows)
print(f"\nWritten: {theta_csv}")

alpha_csv = os.path.join(RESULTS_DIR, "ion_resolved_alpha.csv")
with open(alpha_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["anion","T","ion",
                                        "alpha_soft","alpha_soft_se",
                                        "alpha_hard","alpha_hard_se"])
    w.writeheader(); w.writerows(alpha_rows)
print(f"Written: {alpha_csv}")

import pandas as pd
df_theta = pd.DataFrame(theta_rows)
df_alpha = pd.DataFrame(alpha_rows)
for col in ["alpha_soft","alpha_soft_se","alpha_hard","alpha_hard_se"]:
    df_alpha[col] = pd.to_numeric(df_alpha[col], errors="coerce")

# ── Figure 1: theta per ion, all systems ─────────────────────────────────────
# Layout: 3 anions × 4 temperatures = 12 subplots in a 3×4 grid
fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharey=False)
fig.suptitle(r"Per-ion soft-state fraction $\theta_i$ for each Li$^+$ ion", fontsize=11)

x = np.arange(5)
for row_idx, anion in enumerate(ANIONS):
    for col_idx, T in enumerate(TEMPERATURES):
        ax = axes[row_idx, col_idx]
        sub = df_theta[(df_theta["anion"]==anion) & (df_theta["T"]==T)]
        thetas = sub.sort_values("ion")["theta"].values
        mean_th = np.mean(thetas)

        bars = ax.bar(x, thetas, color=T_COLORS[T], width=0.6, alpha=0.85)
        ax.axhline(mean_th, color="k", ls="--", lw=1.2, alpha=0.7, label=f"mean={mean_th:.3f}")
        ax.set_xticks(x); ax.set_xticklabels([f"Li{i}" for i in range(5)], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{anion.upper()}$^-$ {T_LABELS[T]}", fontsize=8)
        ax.legend(fontsize=6.5, framealpha=0.7)
        if col_idx == 0:
            ax.set_ylabel(r"$\theta_i$", fontsize=9)

plt.tight_layout()
out = os.path.join(FIG_DIR, "ion_resolved_theta.pdf")
plt.savefig(out, bbox_inches="tight"); plt.close()
print(f"Figure: {out}")

# ── Figure 2: state timeline for each anion (one figure per anion) ───────────
# Show the first replica (i=0..4) for all 4 temperatures stacked
# x-axis = time (ps), colors = soft (blue) / hard (red)
TOTAL_STEPS = 100_000   # 100 ns at 1 ps/step

for anion in ANIONS:
    n_T = len(TEMPERATURES)
    fig, axes = plt.subplots(n_T, 5, figsize=(16, n_T * 1.4),
                             sharey=True, sharex=True)
    fig.suptitle(f"State timeline — {anion.upper()}$^-$  (blue=soft, red=hard)",
                 fontsize=11)

    for row_idx, T in enumerate(TEMPERATURES):
        T_str = str(T)
        for ion_i in range(5):
            ax = axes[row_idx, ion_i]
            soft_path = os.path.join(CLASSIFY_DIR, "result", anion, "soft",
                                     T_str, f"{TRAIL}_{ion_i}.txt")
            hard_path = os.path.join(CLASSIFY_DIR, "result", anion, "hard",
                                     T_str, f"{TRAIL}_{ion_i}.txt")

            soft_segs = read_segments(soft_path) if os.path.exists(soft_path) else []
            hard_segs = read_segments(hard_path) if os.path.exists(hard_path) else []

            # Draw hard (background) and soft (foreground) spans
            for s, e in hard_segs:
                ax.axvspan(s, e, ymin=0, ymax=1, color="#FF5555", alpha=0.8, lw=0)
            for s, e in soft_segs:
                ax.axvspan(s, e, ymin=0, ymax=1, color="#5555FF", alpha=0.8, lw=0)

            ax.set_xlim(0, TOTAL_STEPS)
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Li$^+_{ion_i}$", fontsize=8)
            if ion_i == 0:
                ax.set_ylabel(T_LABELS[T], fontsize=7, rotation=90, labelpad=3)
            if row_idx == n_T - 1:
                ax.set_xlabel("time (ps)", fontsize=7)
                ax.tick_params(axis="x", labelsize=6)

    soft_patch = mpatches.Patch(color="#5555FF", label="soft")
    hard_patch = mpatches.Patch(color="#FF5555", label="hard")
    fig.legend(handles=[soft_patch, hard_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG_DIR, f"ion_resolved_timeline_{anion}.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Figure: {out}")

print("\nDone.")
