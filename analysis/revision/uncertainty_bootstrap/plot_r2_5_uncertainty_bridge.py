from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/nas_2/transcendence/revision")
ANALYSIS = BASE / "analysis/uncertainty_bootstrap"
OUT_DIR = ANALYSIS / "outputs"
FIG_DIR = ANALYSIS / "figures"
NOTE_PATH = BASE / "notes/03_r2_other_science/r2_5_all_system_uncertainty_bridge_readout.md"
THETA_CSV = OUT_DIR / "theta_uncertainty_summary.csv"
BRIDGE_CSV = OUT_DIR / "bridging_fraction_summary.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

ANION_COLOR = {"fsi": "#1f77b4", "tfsi": "#ff7f0e", "beti": "#2ca02c"}
ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]

mpl.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fnum(v):
    if v in ("", None, "nan", "NaN"):
        return math.nan
    return float(v)


def read_csv(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_merged():
    theta = read_csv(THETA_CSV)
    bridge = read_csv(BRIDGE_CSV)
    bridge_map = {(r["anion"], int(r["T"])): r for r in bridge}
    rows = []
    for r in theta:
        anion = r["anion"]
        T = int(r["T"])
        b = bridge_map[(anion, T)]
        row = {
            "anion": anion,
            "T": T,
            "label": f"{anion.upper()} {T} K",
            "theta": fnum(r["theta"]),
            "theta_se_table_per_ion": fnum(r["theta_se_table_per_ion"]),
            "theta_se_conservative_max": fnum(r["theta_se_conservative_max"]),
            "theta_relative_se": fnum(r["theta_se_conservative_max"]) / fnum(r["theta"]),
            "theta_ion_min": fnum(r["theta_ion_min"]),
            "theta_ion_max": fnum(r["theta_ion_max"]),
            "theta_timeblock_min": fnum(r["theta_timeblock_min"]),
            "theta_timeblock_max": fnum(r["theta_timeblock_max"]),
            "theta_boot_lo_95": fnum(r["theta_boot_lo_95"]),
            "theta_boot_hi_95": fnum(r["theta_boot_hi_95"]),
            "D_soft_A2ps_original": fnum(r["D_soft_A2ps_original"]),
            "D_soft_se_original": fnum(r["D_soft_se_original"]),
            "D_soft_se_conservative_theta": fnum(r["D_soft_se_conservative_theta"]),
            "any_bridge_fraction": fnum(b["any_bridge_fraction"]),
            "any_bridge_block_min": fnum(b["any_bridge_block_min"]),
            "any_bridge_block_max": fnum(b["any_bridge_block_max"]),
            "pair_shared_fraction_mean": fnum(b["pair_shared_fraction_mean"]),
            "li_bridge_fraction_mean": fnum(b["li_bridge_fraction_mean"]),
            "mean_bridging_anions_per_frame": fnum(b["mean_bridging_anions_per_frame"]),
        }
        rows.append(row)
    rows.sort(key=lambda r: (ANIONS.index(r["anion"]), r["T"]))
    return rows


def write_merged_csv(rows):
    out = OUT_DIR / "uncertainty_bridge_merged_summary.csv"
    with out.open("w", newline="") as fh:
        fields = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def plot_main(rows):
    labels = [r["label"] for r in rows]
    colors = [ANION_COLOR[r["anion"]] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    ax_theta, ax_rel, ax_bridge, ax_scatter = axes.flat

    theta = np.array([r["theta"] for r in rows])
    se = np.array([r["theta_se_conservative_max"] for r in rows])
    ax_theta.bar(x, theta, color=colors, alpha=0.82)
    ax_theta.errorbar(x, theta, yerr=se, fmt="none", ecolor="0.15", capsize=2.4, linewidth=0.9)
    ax_theta.set_ylabel(r"$\theta$")
    ax_theta.grid(True, axis="y", alpha=0.22)
    ax_theta.set_xticks(x)
    ax_theta.set_xticklabels([])

    rel = np.array([r["theta_relative_se"] for r in rows])
    ax_rel.bar(x, rel, color=colors, alpha=0.82)
    ax_rel.axhline(0.25, color="0.4", linestyle="--", linewidth=0.9)
    ax_rel.set_ylabel(r"relative SE of $\theta$")
    ax_rel.grid(True, axis="y", alpha=0.22)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels([])

    any_bridge = np.array([r["any_bridge_fraction"] for r in rows])
    pair_bridge = np.array([r["pair_shared_fraction_mean"] for r in rows])
    ax_bridge.bar(x - 0.18, any_bridge, width=0.36, color="#9467bd", label="any bridged frame")
    ax_bridge.bar(x + 0.18, pair_bridge, width=0.36, color="#8c564b", label="Li-pair shared shell")
    ax_bridge.set_ylabel("bridging fraction")
    ax_bridge.grid(True, axis="y", alpha=0.22)
    ax_bridge.legend(frameon=False, loc="upper right")
    ax_bridge.set_xticks(x)
    ax_bridge.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)

    ax_scatter.scatter(any_bridge, rel, s=58, c=colors, edgecolors="0.15")
    highlight_offsets = {
        ("beti", 298): (-44, 10),
        ("beti", 353): (6, 8),
        ("fsi", 373): (6, -13),
    }
    for i, r in enumerate(rows):
        key = (r["anion"], r["T"])
        if key in highlight_offsets:
            ax_scatter.annotate(
                r["label"],
                (any_bridge[i], rel[i]),
                fontsize=7.5,
                xytext=highlight_offsets[key],
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.35"},
            )
    ax_scatter.set_xlabel("any-bridge fraction")
    ax_scatter.set_ylabel(r"relative SE of $\theta$")
    ax_scatter.grid(True, alpha=0.22)

    fig.tight_layout(pad=0.55)
    path = FIG_DIR / "r2_5_theta_uncertainty_and_bridging"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return path

def plot_ranges(rows):
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    colors = [ANION_COLOR[r["anion"]] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, r in enumerate(rows):
        ax.plot([r["theta_ion_min"], r["theta_ion_max"]], [i + 0.12, i + 0.12], color=colors[i], linewidth=3, alpha=0.85)
        ax.plot([r["theta_timeblock_min"], r["theta_timeblock_max"]], [i - 0.12, i - 0.12], color="0.25", linewidth=3, alpha=0.65)
        ax.scatter([r["theta"]], [i], color="white", edgecolor="black", zorder=3, s=35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel(r"$\theta$")
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", linewidth=3, label="ion-to-ion range (colored by anion)"),
            plt.Line2D([0], [0], color="0.25", linewidth=3, label="20 ns block-mean range"),
            plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linewidth=0, label=r"reported $\theta$"),
        ],
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.84),
    )
    fig.tight_layout(pad=0.55)
    path = FIG_DIR / "r2_5_theta_ion_block_ranges"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return path


def write_note(rows, merged_csv, fig_main, fig_ranges):
    worst_rel = sorted(rows, key=lambda r: r["theta_relative_se"], reverse=True)[:5]
    highest_bridge = sorted(rows, key=lambda r: r["any_bridge_fraction"], reverse=True)[:5]
    rel = np.array([r["theta_relative_se"] for r in rows], dtype=float)
    bridge = np.array([r["any_bridge_fraction"] for r in rows], dtype=float)
    corr = float(np.corrcoef(rel, bridge)[0, 1]) if len(rows) > 2 else math.nan
    beti298 = next(r for r in rows if r["anion"] == "beti" and r["T"] == 298)

    lines = []
    lines.append("# R2-5 All-System Uncertainty and Bridging Comparison")
    lines.append("")
    lines.append("Date: 2026-05-30")
    lines.append("")
    lines.append("Purpose: compare block/ion theta uncertainty and bridging-anion coupling for all 12 systems, rather than discussing BETI 298 K alone.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in [fig_main, fig_ranges]:
        lines.append(f"- `{path.with_suffix('.pdf')}`")
        lines.append(f"- `{path.with_suffix('.png')}`")
    lines.append("")
    lines.append("## Key Readout")
    lines.append("")
    lines.append(f"- Correlation between any-bridge fraction and theta relative SE across 12 systems: {corr:.3f}.")
    lines.append(f"- BETI 298 K: theta = {beti298['theta']:.4f} +/- {beti298['theta_se_conservative_max']:.4f}; ion range = {beti298['theta_ion_min']:.4f}-{beti298['theta_ion_max']:.4f}; block range = {beti298['theta_timeblock_min']:.4f}-{beti298['theta_timeblock_max']:.4f}; any-bridge fraction = {beti298['any_bridge_fraction']:.4f}.")
    lines.append("- Largest relative theta uncertainties:")
    for r in worst_rel:
        lines.append(f"  - {r['label']}: relative SE = {r['theta_relative_se']:.2f}, theta = {r['theta']:.4f}, any-bridge = {r['any_bridge_fraction']:.3f}")
    lines.append("- Largest any-bridge fractions:")
    for r in highest_bridge:
        lines.append(f"  - {r['label']}: any-bridge = {r['any_bridge_fraction']:.3f}, pair-shared = {r['pair_shared_fraction_mean']:.3f}, relative SE = {r['theta_relative_se']:.2f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The all-system comparison separates two issues that were previously easy to conflate. Bridging is non-negligible in several systems and justifies treating five-ion standard errors as lower-bound estimates. However, BETI 298 K has almost no bridged frames in the `pair_check` diagnostic; its large theta uncertainty is driven mainly by rare soft-state sampling and ion-to-ion heterogeneity.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Merged summary CSV: `{merged_csv}`")
    lines.append(f"- Theta uncertainty source: `{THETA_CSV}`")
    lines.append(f"- Bridging source: `{BRIDGE_CSV}`")
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def main():
    rows = load_merged()
    merged_csv = write_merged_csv(rows)
    fig_main = plot_main(rows)
    fig_ranges = plot_ranges(rows)
    write_note(rows, merged_csv, fig_main, fig_ranges)
    print(f"Wrote {merged_csv}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
