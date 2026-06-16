"""
Bridging anion timeline visualization for SI.

For each (anion_system, T), reads pair_check/{i}.txt to find timesteps
where a given anion appears in 2+ Li+ solvation shells simultaneously.

Plots a 10-row timeline (one row per Li+ pair (i,j), i<j), colored when
that pair shares at least one bridging anion.

Also plots the total bridging fraction as a function of time.

Output: paper/Images/sensitivity/bridging_timeline_{anion}_{T}.pdf
        (representative: FSI 298K, FSI 423K, TFSI 298K, BETI 353K)

Usage:
    cd path/to/il_paper/code
    python run_bridging_timeline.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

SOURCE_CODE_ROOT = "/nas_2/transcendence/il_paper/code"
REVISION_ROOT = "/nas_2/transcendence/revision"
CODE_ROOT = SOURCE_CODE_ROOT
sys.path.insert(0, CODE_ROOT)
from config import FIGURE_ROOT

DATA_ROOT = "/nas_2/transcendence/_delete/cowork/my_work"

FIG_DIR   = os.path.join(REVISION_ROOT, "paper", "Images", "sensitivity")
os.makedirs(FIG_DIR, exist_ok=True)

# Systems to plot (cover all three anions + range of temperatures)
PLOT_SYSTEMS = [
    ("fsi",  298),
    ("fsi",  423),
    ("tfsi", 298),
    ("beti", 353),
]

PAIRS = list(combinations(range(5), 2))   # 10 pairs: (0,1),(0,2),...,(3,4)

BLOCK = 100   # sample every BLOCK steps for speed (1 ps × 100 = 100 ps resolution)


def load_shells(anion, T):
    """Load pair_check data: shells[i][t] = set of anion indices in Li_i shell at step t."""
    path = os.path.join(DATA_ROOT, anion, str(T), "pair_check")
    shells = []
    for i in range(5):
        fp = os.path.join(path, f"{i}.txt")
        ion_shells = []
        with open(fp) as fh:
            for line in fh:
                parts = line.strip().rstrip(",").split(",")
                anions_in = set(int(x.strip()) for x in parts[1:] if x.strip())
                ion_shells.append(anions_in)
        shells.append(ion_shells)
    return shells


for anion, T in PLOT_SYSTEMS:
    print(f"Processing {anion} {T}K ...", flush=True)
    shells = load_shells(anion, T)
    n_steps = min(len(s) for s in shells)

    # Sample timesteps
    t_sample = np.arange(0, n_steps, BLOCK)
    n_sample  = len(t_sample)

    # For each pair (i,j) and each sampled timestep: bridging?
    pair_bridge = np.zeros((len(PAIRS), n_sample), dtype=bool)
    total_bridge_count = np.zeros(n_sample, dtype=int)

    for ts_idx, t in enumerate(t_sample):
        sets = [shells[i][t] for i in range(5)]
        # Find bridging anions: in 2+ shells
        union_so_far = set()
        bridging_anions = set()
        for s in sets:
            bridging_anions |= (s & union_so_far)
            union_so_far   |= s
        total_bridge_count[ts_idx] = len(bridging_anions)

        for p_idx, (i, j) in enumerate(PAIRS):
            shared = shells[i][t] & shells[j][t]
            pair_bridge[p_idx, ts_idx] = bool(shared)

    time_ps = t_sample  # each step = 1 ps

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.2),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax_pairs  = axes[0]
    ax_count  = axes[1]

    # Top: pair timeline
    cmap_true  = "#E05555"   # bridging present
    cmap_false = "#DDDDDD"   # no bridging

    pair_labels = [f"Li$_{i}$–Li$_{j}$" for i, j in PAIRS]
    for p_idx in range(len(PAIRS)):
        y = p_idx
        # draw spans
        bridge_arr = pair_bridge[p_idx]
        # Find contiguous blocks
        changes = np.where(np.diff(np.concatenate([[False], bridge_arr, [False]])))[0]
        for k in range(0, len(changes), 2):
            t_start = time_ps[changes[k]]
            t_end   = time_ps[min(changes[k+1]-1, n_sample-1)]
            ax_pairs.axvspan(t_start, t_end, ymin=(y)/len(PAIRS),
                             ymax=(y+1)/len(PAIRS),
                             color=cmap_true, lw=0)

    ax_pairs.set_xlim(0, time_ps[-1])
    ax_pairs.set_ylim(0, len(PAIRS))
    ax_pairs.set_yticks(np.arange(len(PAIRS)) + 0.5)
    ax_pairs.set_yticklabels(pair_labels, fontsize=9)
    ax_pairs.set_ylabel("Li$^+$ pair", fontsize=10)
    ax_pairs.set_xticklabels([])
    ax_pairs.set_facecolor(cmap_false)

    bridge_patch = mpatches.Patch(color=cmap_true, label="bridging anion present")
    no_patch     = mpatches.Patch(color=cmap_false, label="no bridging")
    ax_pairs.legend(
        handles=[bridge_patch, no_patch],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.03),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    # Bottom: total bridging anion count
    ax_count.plot(time_ps, total_bridge_count, color="#333333", lw=0.6, alpha=0.8)
    ax_count.fill_between(time_ps, 0, total_bridge_count, color="#E05555", alpha=0.4)
    ax_count.set_xlim(0, time_ps[-1])
    ax_count.set_ylim(0, total_bridge_count.max() + 0.5)
    ax_count.set_xlabel("Time (ps)", fontsize=10)
    ax_count.set_ylabel("# bridging\nanions", fontsize=9)
    ax_count.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 1], pad=0.5)
    out = os.path.join(FIG_DIR, f"bridging_timeline_{anion}_{T}.pdf")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out}")

print("\nDone.")
