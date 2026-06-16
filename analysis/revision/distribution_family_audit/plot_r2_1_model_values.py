from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/nas_2/transcendence/revision")
AUDIT = BASE / "analysis/distribution_family_audit"
OUT_DIR = AUDIT / "outputs/r2_1_value_plots"
FIG_DIR = AUDIT / "figures"
NOTE_PATH = BASE / "notes/01_r2_1_distribution_statistics/r2_1_value_and_fit_quality_readout.md"
SHORT_FITS = AUDIT / "outputs/short_time_cutoff/short_time_cutoff_model_fits_all.csv"
DIST_STATS = AUDIT / "outputs/distribution_stats.csv"
SLOPES = BASE / "analysis/total_survival_slope_matching/outputs/fixed_split_10ps.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["total_interevent", "total_survival", "state_duration", "state_survival"]
DATASET_LABEL = {
    "total_interevent": "total inter-event",
    "total_survival": "total survival",
    "state_duration": "state duration",
    "state_survival": "state survival",
}
MODELS = [
    "exponential_conditional",
    "weibull_conditional",
    "pareto_power_law",
    "tempered_power_law",
    "biexponential_conditional",
]
MODEL_LABEL = {
    "exponential_conditional": "exp",
    "weibull_conditional": "Weibull",
    "pareto_power_law": "Pareto",
    "tempered_power_law": "tempered",
    "biexponential_conditional": "biexp",
}
MODEL_COLOR = {
    "exponential_conditional": "#777777",
    "weibull_conditional": "#9467bd",
    "pareto_power_law": "#ff7f0e",
    "tempered_power_law": "#2ca02c",
    "biexponential_conditional": "#1f77b4",
}


def fnum(v):
    if v in ("", None, "nan", "NaN"):
        return math.nan
    try:
        return float(v)
    except Exception:
        return math.nan


def read_csv(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def parse_params(s):
    out = {}
    for part in (s or "").split(";"):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        out[key] = fnum(val)
    return out


def load_fits():
    rows = read_csv(SHORT_FITS)
    clean = []
    for row in rows:
        if row["model"] == "all_models":
            continue
        if row.get("ok") != "True":
            continue
        row["T"] = int(row["T"])
        row["tmin"] = fnum(row["tmin"])
        row["n"] = int(float(row["n"]))
        for key in ["AIC", "BIC", "delta_AIC", "delta_BIC", "loglik", "xmin", "xmax_observed"]:
            row[key] = fnum(row[key])
        row["params_dict"] = parse_params(row.get("params", ""))
        clean.append(row)
    return clean


def case_key(row):
    return (row["dataset"], row["anion"], row["T"], row["state"], row["tmin"])


def summarize_model_deltas(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["tmin"], row["model"])].append(row)

    summary = []
    for dataset in DATASETS:
        for tmin in sorted({r["tmin"] for r in rows if r["dataset"] == dataset}):
            for model in MODELS:
                vals = [r["delta_AIC"] for r in groups.get((dataset, tmin, model), []) if np.isfinite(r["delta_AIC"])]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                summary.append(
                    {
                        "dataset": dataset,
                        "tmin": tmin,
                        "model": model,
                        "n_cases": len(arr),
                        "median_delta_AIC": float(np.median(arr)),
                        "q25_delta_AIC": float(np.percentile(arr, 25)),
                        "q75_delta_AIC": float(np.percentile(arr, 75)),
                        "max_delta_AIC": float(np.max(arr)),
                        "best_count": int(np.sum(arr == 0)),
                        "competitive_count_delta_le_2": int(np.sum(arr <= 2)),
                        "near_count_delta_le_10": int(np.sum(arr <= 10)),
                    }
                )

    out = OUT_DIR / "r2_1_model_delta_summary.csv"
    with out.open("w", newline="") as fh:
        fields = list(summary[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    return summary, out


def best_model_counts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[case_key(row)].append(row)

    count_rows = []
    for dataset in DATASETS:
        for tmin in sorted({r["tmin"] for r in rows if r["dataset"] == dataset}):
            subset = {k: v for k, v in groups.items() if k[0] == dataset and k[-1] == tmin}
            best = []
            for _key, vals in subset.items():
                vals = sorted(vals, key=lambda r: r["AIC"])
                if vals:
                    best.append(vals[0]["model"])
            counter = Counter(best)
            row = {"dataset": dataset, "tmin": tmin, "n_cases": len(best)}
            for model in MODELS:
                row[model] = counter.get(model, 0)
            count_rows.append(row)

    out = OUT_DIR / "r2_1_best_model_counts_by_tmin.csv"
    with out.open("w", newline="") as fh:
        fields = ["dataset", "tmin", "n_cases"] + MODELS
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(count_rows)
    return count_rows, out


def parameter_summary(rows):
    param_rows = []
    wanted = [
        ("weibull_conditional", "shape"),
        ("pareto_power_law", "alpha"),
        ("tempered_power_law", "alpha"),
        ("tempered_power_law", "lambda"),
        ("biexponential_conditional", "p"),
        ("biexponential_conditional", "lambda1"),
        ("biexponential_conditional", "lambda2"),
    ]
    for dataset in DATASETS:
        for tmin in sorted({r["tmin"] for r in rows if r["dataset"] == dataset}):
            for model, par in wanted:
                vals = [
                    r["params_dict"].get(par, math.nan)
                    for r in rows
                    if r["dataset"] == dataset and r["tmin"] == tmin and r["model"] == model
                ]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                param_rows.append(
                    {
                        "dataset": dataset,
                        "tmin": tmin,
                        "model": model,
                        "parameter": par,
                        "n_cases": len(arr),
                        "median": float(np.median(arr)),
                        "q25": float(np.percentile(arr, 25)),
                        "q75": float(np.percentile(arr, 75)),
                    }
                )
    out = OUT_DIR / "r2_1_model_parameter_summary.csv"
    with out.open("w", newline="") as fh:
        fields = list(param_rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(param_rows)
    return param_rows, out


def plot_delta_summary(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        sub = [r for r in summary if r["dataset"] == dataset]
        for model in MODELS:
            rows = sorted([r for r in sub if r["model"] == model], key=lambda r: r["tmin"])
            if not rows:
                continue
            x = [r["tmin"] for r in rows]
            y = [max(r["median_delta_AIC"], 1e-3) for r in rows]
            ylo = [max(r["q25_delta_AIC"], 1e-3) for r in rows]
            yhi = [max(r["q75_delta_AIC"], 1e-3) for r in rows]
            ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, color=MODEL_COLOR[model], label=MODEL_LABEL[model])
            ax.fill_between(x, ylo, yhi, color=MODEL_COLOR[model], alpha=0.12, linewidth=0)
        ax.axhline(2, color="0.45", linestyle="--", linewidth=0.9)
        ax.axhline(10, color="0.25", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_xlabel("lower cutoff tmin (ps)")
        ax.set_ylabel("median delta AIC")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("R2-1 model-selection values across short-time cutoffs", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "r2_1_delta_aic_vs_tmin"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return path


def plot_best_counts(count_rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        rows = sorted([r for r in count_rows if r["dataset"] == dataset], key=lambda r: r["tmin"])
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for model in MODELS:
            y = np.array([r[model] for r in rows], dtype=float)
            ax.bar(x, y, bottom=bottom, color=MODEL_COLOR[model], label=MODEL_LABEL[model])
            bottom += y
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r["tmin"])) for r in rows])
        ax.set_xlabel("tmin (ps)")
        ax.set_ylabel("best-AIC count")
        ax.grid(True, axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("R2-1 best model counts by cutoff", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "r2_1_best_model_counts_vs_tmin"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return path


def plot_distribution_stats():
    rows = read_csv(DIST_STATS)
    clean = []
    for row in rows:
        if row["dataset"] not in DATASETS:
            continue
        row["cv"] = fnum(row["cv"])
        row["burstiness_A_N"] = fnum(row["burstiness_A_N"])
        row["q99_over_median"] = fnum(row["q99_over_median"])
        clean.append(row)

    metrics = [("cv", "coefficient of variation"), ("burstiness_A_N", "A_N"), ("q99_over_median", "q99 / median")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    positions = np.arange(len(DATASETS))
    for ax, (metric, label) in zip(axes, metrics):
        data = []
        for dataset in DATASETS:
            vals = [r[metric] for r in clean if r["dataset"] == dataset and np.isfinite(r[metric])]
            data.append(vals)
        ax.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        for i, vals in enumerate(data):
            jitter = np.linspace(-0.16, 0.16, len(vals)) if vals else []
            ax.scatter(np.asarray(jitter) + i, vals, s=14, alpha=0.7, color="#1f77b4")
        ax.set_xticks(positions)
        ax.set_xticklabels([DATASET_LABEL[d].replace(" ", "\n") for d in DATASETS], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.22)
        if metric == "q99_over_median":
            ax.set_yscale("log")
    fig.suptitle("Actual distribution-shape values behind R2-1")
    fig.tight_layout()
    path = FIG_DIR / "r2_1_distribution_shape_values"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return path


def plot_slope_values():
    rows = read_csv(SLOPES)
    labels = []
    beta_total = []
    beta_soft = []
    beta_hard = []
    r2_total = []
    r2_soft = []
    r2_hard = []
    for row in rows:
        labels.append(f"{row['anion'].upper()}\n{row['T']}")
        beta_total.append(fnum(row["beta_total_early10"]))
        beta_soft.append(fnum(row["beta_soft_full"]))
        beta_hard.append(fnum(row["beta_hard_full"]))
        r2_total.append(fnum(row["r2_total_early10"]))
        r2_soft.append(fnum(row["r2_soft_full"]))
        r2_hard.append(fnum(row["r2_hard_full"]))

    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    axes[0].bar(x - width, beta_soft, width=width, label="soft beta", color="#1f77b4")
    axes[0].bar(x, beta_hard, width=width, label="hard beta", color="#d62728")
    axes[0].bar(x + width, beta_total, width=width, label="total beta", color="#555555")
    axes[0].set_ylabel("descriptive beta")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(True, axis="y", alpha=0.22)

    axes[1].plot(x, r2_soft, marker="o", label="soft R2", color="#1f77b4")
    axes[1].plot(x, r2_hard, marker="s", label="hard R2", color="#d62728")
    axes[1].plot(x, r2_total, marker="^", label="total R2", color="#555555")
    axes[1].axhline(0.9, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("log-log fit R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(True, axis="y", alpha=0.22)
    fig.suptitle("Slope-ordering values and descriptive fit quality")
    fig.tight_layout()
    path = FIG_DIR / "r2_1_slope_ordering_beta_r2"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return path


def write_note(summary, count_rows, delta_csv, counts_csv, param_csv, figure_paths):
    exp_summary = [r for r in summary if r["model"] == "exponential_conditional"]
    temp_summary = [r for r in summary if r["model"] == "tempered_power_law"]
    lines = []
    lines.append("# R2-1 Actual Values and Fit-Quality Readout")
    lines.append("")
    lines.append("Date: 2026-05-30")
    lines.append("")
    lines.append("Purpose: supplement the earlier pass/fail model-count screening with actual values: delta AIC magnitudes, cutoff dependence, best-model counts, distribution-shape statistics, and slope/R2 values.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        lines.append(f"- `{path.with_suffix('.pdf')}`")
        lines.append(f"- `{path.with_suffix('.png')}`")
    lines.append("")
    lines.append("## How To Read The Plots")
    lines.append("")
    lines.append("- delta AIC = AIC(model) - AIC(best model for the same data/window).")
    lines.append("- delta AIC = 0 means best in that comparison.")
    lines.append("- delta AIC <= 2 is treated as competitive; delta AIC > 10 is treated as weakly supported relative to the best model.")
    lines.append("- The cutoff plots show how conclusions change when events below 1, 2, 5, 10, or 50 ps are excluded.")
    lines.append("")
    lines.append("## Key Numerical Summary")
    lines.append("")
    for dataset in DATASETS:
        e2 = next((r for r in exp_summary if r["dataset"] == dataset and r["tmin"] == 2.0), None)
        t2 = next((r for r in temp_summary if r["dataset"] == dataset and r["tmin"] == 2.0), None)
        if e2:
            lines.append(
                f"- {DATASET_LABEL[dataset]}, tmin = 2 ps: median delta AIC for single exponential = {e2['median_delta_AIC']:.1f}; "
                f"competitive cases = {e2['competitive_count_delta_le_2']}/{e2['n_cases']}."
            )
        if t2:
            lines.append(
                f"  Tempered power law at the same cutoff: best cases = {t2['best_count']}/{t2['n_cases']}, "
                f"competitive cases = {t2['competitive_count_delta_le_2']}/{t2['n_cases']}."
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This analysis is the graph-based version of the R2-1 screen. It shows the magnitude of model preference, not just which model won. Total inter-event and total survival are strongly inconsistent with a single exponential by very large delta AIC values. State durations are much closer to single-rate behavior, but not exactly exponential over the full raw range. State-resolved survival remains non-single-exponential and composite; power-law-like descriptors should therefore be presented as finite-window descriptors rather than universal pure power laws.")
    lines.append("")
    lines.append("## Output Data")
    lines.append("")
    lines.append(f"- delta-AIC summary: `{delta_csv}`")
    lines.append(f"- best-model counts: `{counts_csv}`")
    lines.append(f"- fitted-parameter summary: `{param_csv}`")
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def main():
    rows = load_fits()
    summary, delta_csv = summarize_model_deltas(rows)
    count_rows, counts_csv = best_model_counts(rows)
    param_rows, param_csv = parameter_summary(rows)
    fig1 = plot_delta_summary(summary)
    fig2 = plot_best_counts(count_rows)
    fig3 = plot_distribution_stats()
    fig4 = plot_slope_values()
    write_note(summary, count_rows, delta_csv, counts_csv, param_csv, [fig1, fig2, fig3, fig4])
    print(f"Wrote {delta_csv}")
    print(f"Wrote {counts_csv}")
    print(f"Wrote {param_csv}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote {NOTE_PATH}")


if __name__ == "__main__":
    main()
