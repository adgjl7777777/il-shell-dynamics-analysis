from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BASE = Path("/nas_2/transcendence/revision")
CSV_PATH = BASE / "analysis/threshold_sensitivity_extended/outputs/threshold_sensitivity_extended.csv"
FIG_DIR = BASE / "analysis/threshold_sensitivity_extended/figures"
OUT_DIR = BASE / "analysis/threshold_sensitivity_extended/outputs"
NOTE_PATH = BASE / "notes/03_r2_other_science/threshold_full_range_readout.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]
COLORS = {"soft": "#1f77b4", "hard": "#d62728", "theta": "#222222", "mobility": "#2ca02c"}
ANION_LABELS = {"fsi": r"FSI$^-$", "tfsi": r"TFSI$^-$", "beti": r"BETI$^-$"}
ANION_COLORS = {"fsi": "#1f77b4", "tfsi": "#ff7f0e", "beti": "#2ca02c"}
TEMP_MARKERS = {298: "o", 353: "^", 373: "s", 423: "D"}

mpl.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fnum(v):
    if v in (None, "", "nan", "NaN"):
        return math.nan
    try:
        return float(v)
    except Exception:
        return math.nan


def read_rows():
    with CSV_PATH.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k not in {"anion"}:
                r[k] = fnum(v)
        r["T"] = int(r["T"])
    return rows


def by_system(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r["anion"], r["T"])].append(r)
    for key in d:
        d[key].sort(key=lambda x: x["trail"])
    return d


def system_axes(ylabel, outfile, plotter, yscale="linear"):
    rows = read_rows()
    d = by_system(rows)
    fig, axes = plt.subplots(len(ANIONS), len(TEMPS), figsize=(7.2, 6.25), sharex=True, sharey=False)
    for i, anion in enumerate(ANIONS):
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            sub = d[(anion, T)]
            plotter(ax, sub, anion, T)
            ax.set_xscale("log")
            if yscale == "log":
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
            ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0)
            if i == 0:
                ax.text(0.05, 0.94, f"{T} K", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2})
            if j == 0:
                ax.set_ylabel(f"{ANION_LABELS[anion]}\n{ylabel}", fontsize=9.5)
            ax.set_xticks([0.1, 1.0, 10.0])
            ax.set_xticklabels(["0.1", "1", "10"])
            ax.tick_params(labelsize=8, length=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False, fontsize=8.8)
    fig.supxlabel(r"threshold multiplier $\eta$", fontsize=10)
    fig.tight_layout(rect=[0.02, 0.045, 0.995, 0.95], pad=0.45)
    fig.savefig(outfile.with_suffix(".pdf"))
    fig.savefig(outfile.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_theta(ax, sub, anion, T):
    x = np.array([r["trail"] for r in sub], dtype=float)
    y = np.array([r["theta"] for r in sub], dtype=float)
    err = np.array([r["theta_se"] for r in sub], dtype=float)
    ax.errorbar(x, y, yerr=err, color=COLORS["theta"], marker="o", markersize=3.0, linewidth=1.15, capsize=1.5, label=r"$\theta$")
    ax.set_ylim(bottom=0)


def plot_alpha(ax, sub, anion, T):
    x = np.array([r["trail"] for r in sub], dtype=float)
    ys = np.array([r["alpha_soft"] for r in sub], dtype=float)
    yh = np.array([r["alpha_hard"] for r in sub], dtype=float)
    ax.plot(x, ys, color=COLORS["soft"], marker="o", markersize=3.0, linewidth=1.15, label=r"$\alpha(h_\mathrm{soft})$")
    ax.plot(x, yh, color=COLORS["hard"], marker="s", markersize=3.0, linewidth=1.15, label=r"$\alpha(h_\mathrm{hard})$")
    finite = np.concatenate([ys[np.isfinite(ys) & (ys > 0)], yh[np.isfinite(yh) & (yh > 0)]])
    if len(finite):
        ax.set_ylim(max(finite.min() * 0.6, 1e-5), finite.max() * 1.8)


def plot_burstiness(ax, sub, anion, T):
    x = np.array([r["trail"] for r in sub], dtype=float)
    ys = np.array([r["A_shell_soft"] for r in sub], dtype=float)
    yh = np.array([r["A_shell_hard"] for r in sub], dtype=float)
    ax.axhline(0, color="0.55", linestyle="--", linewidth=0.8)
    ax.plot(x, ys, color=COLORS["soft"], marker="o", markersize=3.0, linewidth=1.15, label=r"$A_N$ soft")
    ax.plot(x, yh, color=COLORS["hard"], marker="s", markersize=3.0, linewidth=1.15, label=r"$A_N$ hard")
    ax.set_ylim(-0.45, 0.65)


def plot_mobility(ax, sub, anion, T):
    x = np.array([r["trail"] for r in sub], dtype=float)
    y = np.array([r["D_soft_over_D_hard_theta_only"] for r in sub], dtype=float)
    ax.axhline(1, color="0.55", linestyle="--", linewidth=0.8)
    ax.plot(x, y, color=COLORS["mobility"], marker="o", markersize=3.0, linewidth=1.15, label=r"$D_\mathrm{soft}/D_\mathrm{hard}$")
    finite = y[np.isfinite(y) & (y > 0)]
    if len(finite):
        ax.set_ylim(max(0.8, finite.min() * 0.75), finite.max() * 1.4)


def plot_combined_overview(rows):
    d = by_system(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=True)
    ax_theta, ax_alpha, ax_burst, ax_mob = axes.ravel()

    for anion in ANIONS:
        for T in TEMPS:
            sub = d[(anion, T)]
            x = np.array([r["trail"] for r in sub], dtype=float)
            common = {
                "color": ANION_COLORS[anion],
                "marker": TEMP_MARKERS[T],
                "markersize": 2.4,
                "markevery": 6,
                "linewidth": 0.9,
                "alpha": 0.76,
            }
            ax_theta.plot(x, [r["theta"] for r in sub], **common)
            ax_alpha.plot(x, [r["alpha_soft"] for r in sub], linestyle="-", **common)
            ax_alpha.plot(x, [r["alpha_hard"] for r in sub], linestyle="--", **common)
            ax_burst.plot(x, [r["A_shell_soft"] for r in sub], linestyle="-", **common)
            ax_burst.plot(x, [r["A_shell_hard"] for r in sub], linestyle="--", **common)
            ax_mob.plot(x, [r["D_soft_over_D_hard_theta_only"] for r in sub], **common)

    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1.0, 10.0])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=8, length=3)

    ax_alpha.set_yscale("log")
    ax_mob.set_yscale("log")
    ax_burst.axhline(0, color="0.55", linestyle=":", linewidth=0.8)
    ax_mob.axhline(1, color="0.55", linestyle=":", linewidth=0.8)

    ax_theta.set_ylabel(r"$\theta$")
    ax_alpha.set_ylabel(r"$\alpha(h)$")
    ax_burst.set_ylabel(r"$A_N$")
    ax_mob.set_ylabel(r"$D_\mathrm{soft}/D_\mathrm{hard}$")
    ax_burst.set_xlabel(r"threshold multiplier $\eta$")
    ax_mob.set_xlabel(r"threshold multiplier $\eta$")

    ax_theta.text(0.03, 0.95, "soft fraction", transform=ax_theta.transAxes, va="top", ha="left", fontsize=9)
    ax_alpha.text(0.03, 0.95, "transition rates", transform=ax_alpha.transAxes, va="top", ha="left", fontsize=9)
    ax_burst.text(0.03, 0.95, "burstiness", transform=ax_burst.transAxes, va="top", ha="left", fontsize=9)
    ax_mob.text(0.03, 0.95, "mobility ratio", transform=ax_mob.transAxes, va="top", ha="left", fontsize=9)

    anion_handles = [Line2D([0], [0], color=ANION_COLORS[a], lw=1.4, label=ANION_LABELS[a]) for a in ANIONS]
    state_handles = [
        Line2D([0], [0], color="0.25", lw=1.2, linestyle="-", label="soft"),
        Line2D([0], [0], color="0.25", lw=1.2, linestyle="--", label="hard"),
    ]
    temp_handles = [Line2D([0], [0], color="0.25", marker=TEMP_MARKERS[T], linestyle="None", markersize=4.0, label=f"{T} K") for T in TEMPS]
    fig.legend(handles=anion_handles + state_handles + temp_handles, loc="upper center", ncol=9, frameon=False, fontsize=7.4)
    fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.91], pad=0.75)
    outfile = FIG_DIR / "threshold_sensitivity_overview_2x2"
    fig.savefig(outfile.with_suffix(".pdf"))
    fig.savefig(outfile.with_suffix(".png"), dpi=240)
    plt.close(fig)


def write_summary(rows):
    system_summary = []
    for anion in ANIONS:
        for T in TEMPS:
            sub = sorted([r for r in rows if r["anion"] == anion and r["T"] == T], key=lambda r: r["trail"])
            n = len(sub)
            theta = [r["theta"] for r in sub if np.isfinite(r["theta"])]
            mobility = [r["D_soft_over_D_hard_theta_only"] for r in sub if np.isfinite(r["D_soft_over_D_hard_theta_only"])]
            shell_gt = [r for r in sub if np.isfinite(r["A_shell_soft"]) and np.isfinite(r["A_shell_hard"]) and r["A_shell_soft"] > r["A_shell_hard"]]
            alpha_soft_finite = [r for r in sub if np.isfinite(r["alpha_soft"])]
            alpha_hard_finite = [r for r in sub if np.isfinite(r["alpha_hard"])]
            system_summary.append({
                "anion": anion,
                "T": T,
                "n_eta": n,
                "theta_min": min(theta),
                "theta_max": max(theta),
                "A_shell_soft_gt_hard_count": len(shell_gt),
                "A_shell_soft_gt_hard_fraction": len(shell_gt) / n,
                "D_ratio_min": min(mobility),
                "D_ratio_max": max(mobility),
                "D_ratio_gt_1_count": sum(v > 1 for v in mobility),
                "alpha_soft_finite_count": len(alpha_soft_finite),
                "alpha_hard_finite_count": len(alpha_hard_finite),
            })

    summary_csv = OUT_DIR / "threshold_full_range_system_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        fields = list(system_summary[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(system_summary)

    n_total = len(rows)
    d_gt = sum(r["D_soft_over_D_hard_theta_only"] > 1 for r in rows if np.isfinite(r["D_soft_over_D_hard_theta_only"]))
    a_gt = sum(r["A_shell_soft"] > r["A_shell_hard"] for r in rows if np.isfinite(r["A_shell_soft"]) and np.isfinite(r["A_shell_hard"]))
    finite_a = sum(1 for r in rows if np.isfinite(r["A_shell_soft"]) and np.isfinite(r["A_shell_hard"]))
    worst_shell = sorted(system_summary, key=lambda r: r["A_shell_soft_gt_hard_fraction"])[:4]
    min_d = min(system_summary, key=lambda r: r["D_ratio_min"])
    max_d = max(system_summary, key=lambda r: r["D_ratio_max"])

    lines = [
        "# Full-Range Threshold Sensitivity Readout",
        "",
        "Date: 2026-05-31",
        "",
        "Scope: eta = 0.1 to 10.0 for all 12 anion-temperature systems, using existing threshold-resolved classifications.",
        "",
        "## Output Figures",
        "",
    ]
    for name in [
        "threshold_sensitivity_overview_2x2",
        "threshold_theta_curves",
        "threshold_alpha_curves",
        "threshold_burstiness_shell_curves",
        "threshold_mobility_ratio_curves",
    ]:
        lines.append(f"- `{FIG_DIR / (name + '.pdf')}`")
        lines.append(f"- `{FIG_DIR / (name + '.png')}`")
    lines.extend([
        "",
        "## Main Numerical Readout",
        "",
        f"- Full grid size: {n_total} system-threshold rows.",
        f"- D_soft/D_hard > 1 across the full eta range: {d_gt}/{n_total} rows.",
        f"- A_shell,soft > A_shell,hard across finite rows: {a_gt}/{finite_a} rows.",
        "- Smallest D_soft/D_hard occurs for {anion} {T} K: min = {val:.3f}.".format(anion=min_d["anion"].upper(), T=min_d["T"], val=min_d["D_ratio_min"]),
        "- Largest D_soft/D_hard occurs for {anion} {T} K: max = {val:.3f}.".format(anion=max_d["anion"].upper(), T=max_d["T"], val=max_d["D_ratio_max"]),
        "- Lowest fractions of eta values with A_shell,soft > A_shell,hard:",
    ])
    for r in worst_shell:
        lines.append("  - {anion} {T} K: {count}/{n} eta values".format(anion=r["anion"].upper(), T=r["T"], count=r["A_shell_soft_gt_hard_count"], n=r["n_eta"]))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The full threshold scan supports plotting the actual threshold dependence instead of only reporting pass/fail counts at eta = 0.5, 1.0, and 2.0. The new 2x2 overview provides a compact reviewer-facing SI figure, while the detailed 3x4 figures remain as generated analysis outputs. The mobility ratio is the most stable quantity across eta. The burstiness contrast is more threshold-sensitive, especially at permissive small eta, so the manuscript should avoid claiming strict threshold independence of A_shell.",
        "",
        "## Data Files",
        "",
        f"- Full row CSV: `{CSV_PATH}`",
        f"- System summary CSV: `{summary_csv}`",
    ])
    NOTE_PATH.write_text("\n".join(lines) + "\n")
    return summary_csv


def main():
    rows = read_rows()
    plot_combined_overview(rows)
    system_axes(r"$\theta$", FIG_DIR / "threshold_theta_curves", plot_theta)
    system_axes(r"$\alpha(h)$", FIG_DIR / "threshold_alpha_curves", plot_alpha, yscale="log")
    system_axes(r"$A_N$", FIG_DIR / "threshold_burstiness_shell_curves", plot_burstiness)
    system_axes(r"$D_\mathrm{soft}/D_\mathrm{hard}$", FIG_DIR / "threshold_mobility_ratio_curves", plot_mobility, yscale="log")
    summary_csv = write_summary(rows)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
