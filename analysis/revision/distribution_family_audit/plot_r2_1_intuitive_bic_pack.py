#!/usr/bin/env python3
"""Make intuitive AIC/BIC and two-state justification plots for R2-1.

All inputs are existing revision analysis outputs. New figures and summary
tables are written under the revision workspace.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REV = Path("/nas_2/transcendence/revision")
AUDIT = REV / "analysis" / "distribution_family_audit"
SHORT_FITS = AUDIT / "outputs" / "short_time_cutoff" / "short_time_cutoff_model_fits_all.csv"
SLOPE_CSV = (
    REV
    / "analysis"
    / "total_survival_slope_matching"
    / "outputs"
    / "fixed_split_10ps.csv"
)

FIG_DIR = AUDIT / "figures"
OUT_DIR = AUDIT / "outputs" / "r2_1_intuitive_bic"


DATASET_LABELS = {
    "total_interevent": "Total inter-event",
    "total_survival": "Total pair survival",
    "state_duration": "State residence duration",
    "state_survival": "Intra-state survival",
}

DATASET_ORDER = [
    "total_interevent",
    "total_survival",
    "state_duration",
    "state_survival",
]

MODEL_LABELS = {
    "exponential_conditional": "exponential",
    "weibull_conditional": "Weibull",
    "pareto_power_law": "power law",
    "tempered_power_law": "tempered power",
    "biexponential_conditional": "biexponential",
}

MODEL_ORDER = [
    "exponential_conditional",
    "weibull_conditional",
    "pareto_power_law",
    "tempered_power_law",
    "biexponential_conditional",
]

MODEL_COLORS = {
    "exponential_conditional": "#6f7782",
    "weibull_conditional": "#7b61a8",
    "pareto_power_law": "#c9792c",
    "tempered_power_law": "#2f8f6b",
    "biexponential_conditional": "#3f6fb5",
}

ANION_ORDER = {"fsi": 0, "tfsi": 1, "beti": 2}
STATE_ORDER = {"total": 0, "soft": 1, "hard": 2}


def clean_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, str(model))


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def savefig(
    fig: plt.Figure,
    stem: str,
    rect: tuple[float, float, float, float] = (0.02, 0.02, 0.98, 0.98),
) -> None:
    fig.tight_layout(rect=rect, pad=0.45)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def criterion_tag(ax: plt.Axes, criterion: str) -> None:
    ax.text(
        0.97,
        0.96,
        criterion,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def criterion_delta_col(criterion: str) -> str:
    return f"delta_{criterion}"


def finite_fit_table() -> pd.DataFrame:
    df = pd.read_csv(SHORT_FITS)
    df = df[(df["ok"] == True) & df["model"].isin(MODEL_ORDER)].copy()  # noqa: E712
    df["anion_order"] = df["anion"].map(ANION_ORDER)
    df["state_order"] = df["state"].map(STATE_ORDER)
    df = df.sort_values(["dataset", "anion_order", "T", "state_order", "tmin", "model"])
    return df


def case_label(row: pd.Series) -> str:
    base = f"{str(row['anion']).upper()} {int(row['T'])} K"
    if row["state"] == "total":
        return base
    return f"{base} {row['state']}"


def make_delta_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for criterion in ["AIC", "BIC"]:
        col = criterion_delta_col(criterion)
        for keys, sub in df.groupby(["dataset", "tmin", "model"], sort=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            rows.append(
                {
                    "criterion": criterion,
                    "dataset": keys[0],
                    "tmin": keys[1],
                    "model": keys[2],
                    "n_cases": int(vals.size),
                    "median_delta": float(np.median(vals)),
                    "q25_delta": float(np.quantile(vals, 0.25)),
                    "q75_delta": float(np.quantile(vals, 0.75)),
                    "n_competitive_delta_le_2": int(np.sum(vals <= 2.0)),
                    "n_unsupported_delta_gt_10": int(np.sum(vals > 10.0)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r2_1_aic_bic_delta_summary.csv", index=False)
    return out


def make_best_model_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_cols = ["dataset", "anion", "T", "state", "tmin"]
    case_rows = []
    count_rows = []
    for criterion in ["AIC", "BIC"]:
        for keys, sub in df.groupby(case_cols, sort=False):
            metric = pd.to_numeric(sub[criterion], errors="coerce")
            best_idx = metric.idxmin()
            best = sub.loc[best_idx]
            delta_col = criterion_delta_col(criterion)
            second_delta = (
                pd.to_numeric(sub[delta_col], errors="coerce")
                .sort_values()
                .iloc[1]
                if len(sub) > 1
                else np.nan
            )
            case_rows.append(
                {
                    "criterion": criterion,
                    "dataset": keys[0],
                    "anion": keys[1],
                    "T": keys[2],
                    "state": keys[3],
                    "tmin": keys[4],
                    "best_model": best["model"],
                    "second_delta": second_delta,
                    "n": int(best["n"]),
                    "case_label": case_label(best),
                }
            )
        case_df_tmp = pd.DataFrame([r for r in case_rows if r["criterion"] == criterion])
        counts = (
            case_df_tmp.groupby(["criterion", "dataset", "tmin", "best_model"], sort=False)
            .size()
            .reset_index(name="n_cases")
        )
        count_rows.append(counts)
    case_df = pd.DataFrame(case_rows)
    count_df = pd.concat(count_rows, ignore_index=True)
    case_df.to_csv(OUT_DIR / "r2_1_case_best_model_aic_bic.csv", index=False)
    count_df.to_csv(OUT_DIR / "r2_1_best_model_counts_aic_bic.csv", index=False)
    return case_df, count_df


def set_delta_axis(ax: plt.Axes) -> None:
    ax.set_yscale("symlog", linthresh=2.0, linscale=0.8)
    ax.set_ylim(bottom=0)
    ax.axhline(2, color="#555555", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(10, color="#222222", lw=0.8, ls=":", alpha=0.75)


def plot_delta_vs_tmin(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=2,
        figsize=(9.0, 10.6),
        sharex=True,
        sharey="row",
    )
    for i, dataset in enumerate(DATASET_ORDER):
        for j, criterion in enumerate(["AIC", "BIC"]):
            ax = axes[i, j]
            sub = summary[(summary["dataset"] == dataset) & (summary["criterion"] == criterion)]
            for model in MODEL_ORDER:
                m = sub[sub["model"] == model].sort_values("tmin")
                if m.empty:
                    continue
                x = m["tmin"].to_numpy(dtype=float)
                y = m["median_delta"].to_numpy(dtype=float)
                q25 = m["q25_delta"].to_numpy(dtype=float)
                q75 = m["q75_delta"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    ms=3.5,
                    lw=1.3,
                    color=MODEL_COLORS[model],
                    label=clean_model_name(model),
                )
                ax.fill_between(x, q25, q75, color=MODEL_COLORS[model], alpha=0.12, lw=0)
            set_delta_axis(ax)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 50])
            ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter("{x:g}"))
            criterion_tag(ax, criterion)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\nmedian Delta score")
            if i == len(DATASET_ORDER) - 1:
                ax.set_xlabel("lower cutoff tmin (ps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
    )
    savefig(fig, "r2_1_aic_bic_delta_vs_tmin", rect=(0.02, 0.02, 0.98, 0.955))


def plot_exponential_rejection(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df["model"] == "exponential_conditional"].copy()
    rows = []
    for criterion in ["AIC", "BIC"]:
        col = criterion_delta_col(criterion)
        for keys, sub in exp.groupby(["dataset", "tmin"], sort=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            rows.append(
                {
                    "criterion": criterion,
                    "dataset": keys[0],
                    "tmin": keys[1],
                    "n_cases": int(vals.size),
                    "median_delta_exponential": float(np.median(vals)),
                    "q25_delta_exponential": float(np.quantile(vals, 0.25)),
                    "q75_delta_exponential": float(np.quantile(vals, 0.75)),
                    "n_competitive_delta_le_2": int(np.sum(vals <= 2.0)),
                    "n_unsupported_delta_gt_10": int(np.sum(vals > 10.0)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "r2_1_exponential_rejection_summary.csv", index=False)

    rng = np.random.default_rng(20260530)
    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        2,
        figsize=(8.6, 10.4),
        sharex=True,
        sharey="row",
    )
    tmins = [1, 2, 5, 10, 50]
    for i, dataset in enumerate(DATASET_ORDER):
        for j, criterion in enumerate(["AIC", "BIC"]):
            ax = axes[i, j]
            col = criterion_delta_col(criterion)
            sub = exp[(exp["dataset"] == dataset)].copy()
            values = [
                pd.to_numeric(sub[sub["tmin"] == t][col], errors="coerce").dropna().to_numpy()
                for t in tmins
            ]
            ax.boxplot(
                values,
                positions=np.arange(len(tmins)),
                widths=0.52,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#111111", "lw": 1.2},
                boxprops={"facecolor": "#d8dde5", "edgecolor": "#4a5564", "lw": 0.8},
                whiskerprops={"color": "#4a5564", "lw": 0.8},
                capprops={"color": "#4a5564", "lw": 0.8},
            )
            for pos, vals in enumerate(values):
                if vals.size == 0:
                    continue
                x = pos + rng.normal(0, 0.045, vals.size)
                ax.scatter(x, vals, s=11, color="#385f8a", alpha=0.65, linewidths=0)
            set_delta_axis(ax)
            ax.set_xticks(np.arange(len(tmins)))
            ax.set_xticklabels([str(t) for t in tmins])
            criterion_tag(ax, criterion)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABELS[dataset]}\nexponential Delta")
            if i == len(DATASET_ORDER) - 1:
                ax.set_xlabel("lower cutoff tmin (ps)")
    savefig(fig, "r2_1_exponential_rejection_aic_bic")
    return summary


def model_code_map() -> dict[str, int]:
    return {m: i for i, m in enumerate(MODEL_ORDER)}


def plot_best_model_heatmaps(case_df: pd.DataFrame) -> None:
    codes = model_code_map()
    cmap = mpl.colors.ListedColormap([MODEL_COLORS[m] for m in MODEL_ORDER])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(MODEL_ORDER) + 0.5, 1), cmap.N)
    tmins = [1, 2, 5, 10, 50]

    for dataset in DATASET_ORDER:
        dataset_cases = case_df[case_df["dataset"] == dataset].copy()
        labels = (
            dataset_cases[["anion", "T", "state", "case_label"]]
            .drop_duplicates()
            .sort_values(
                by=["anion", "T", "state"],
                key=lambda s: s.map(ANION_ORDER).fillna(s)
                if s.name == "anion"
                else s.map(STATE_ORDER).fillna(s)
                if s.name == "state"
                else s,
            )
        )
        row_labels = labels["case_label"].tolist()
        nrows = len(row_labels)
        fig_height = max(4.0, 0.26 * nrows + 1.6)
        fig, axes = plt.subplots(1, 2, figsize=(8.2, fig_height), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for j, criterion in enumerate(["AIC", "BIC"]):
            ax = axes[j]
            mat = np.full((nrows, len(tmins)), np.nan)
            sub = dataset_cases[dataset_cases["criterion"] == criterion]
            for r, label in enumerate(row_labels):
                for c, tmin in enumerate(tmins):
                    hit = sub[(sub["case_label"] == label) & (sub["tmin"] == tmin)]
                    if hit.empty:
                        continue
                    mat[r, c] = codes.get(hit.iloc[0]["best_model"], np.nan)
            masked = np.ma.masked_invalid(mat)
            ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(np.arange(len(tmins)))
            ax.set_xticklabels([str(t) for t in tmins])
            ax.set_xlabel(f"{criterion} tmin (ps)")
            ax.set_yticks(np.arange(nrows))
            ax.set_yticklabels(row_labels if j == 0 else [])
            ax.tick_params(axis="both", length=0)
            ax.set_xticks(np.arange(-0.5, len(tmins), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
            ax.grid(which="minor", color="white", lw=0.8)
            ax.grid(which="major", visible=False)
        handles = [
            mpl.patches.Patch(color=MODEL_COLORS[m], label=clean_model_name(m))
            for m in MODEL_ORDER
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
        )
        savefig(fig, f"r2_1_best_model_heatmap_{dataset}", rect=(0.02, 0.02, 0.98, 0.90))


def plot_best_model_counts(count_df: pd.DataFrame) -> None:
    tmins = [1, 2, 5, 10, 50]
    fig, axes = plt.subplots(
        nrows=len(DATASET_ORDER),
        ncols=2,
        figsize=(8.8, 10.0),
        sharex=True,
    )
    for i, dataset in enumerate(DATASET_ORDER):
        for j, criterion in enumerate(["AIC", "BIC"]):
            ax = axes[i, j]
            bottom = np.zeros(len(tmins), dtype=float)
            for model in MODEL_ORDER:
                vals = []
                for tmin in tmins:
                    hit = count_df[
                        (count_df["dataset"] == dataset)
                        & (count_df["criterion"] == criterion)
                        & (count_df["tmin"] == tmin)
                        & (count_df["best_model"] == model)
                    ]
                    vals.append(float(hit["n_cases"].sum()) if not hit.empty else 0.0)
                ax.bar(
                    np.arange(len(tmins)),
                    vals,
                    bottom=bottom,
                    color=MODEL_COLORS[model],
                    width=0.72,
                    edgecolor="white",
                    linewidth=0.4,
                    label=clean_model_name(model),
                )
                bottom += np.array(vals)
            criterion_tag(ax, criterion)
            ax.set_xticks(np.arange(len(tmins)))
            ax.set_xticklabels([str(t) for t in tmins])
            ax.set_ylabel(f"{DATASET_LABELS[dataset]}\nbest-model count")
            if i == len(DATASET_ORDER) - 1:
                ax.set_xlabel("lower cutoff tmin (ps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
    )
    savefig(fig, "r2_1_best_model_counts_aic_bic", rect=(0.02, 0.02, 0.98, 0.95))


def plot_slope_ladder() -> pd.DataFrame:
    slopes = pd.read_csv(SLOPE_CSV)
    slopes["anion_order"] = slopes["anion"].map(ANION_ORDER)
    slopes = slopes.sort_values(["anion_order", "T"]).copy()
    slopes["label"] = slopes["anion"].str.upper() + " " + slopes["T"].astype(int).astype(str) + " K"
    rows = []
    for _, row in slopes.iterrows():
        rows.append(
            {
                "anion": row["anion"],
                "T": row["T"],
                "beta_total_early10": row["beta_total_early10"],
                "beta_total_late10": row["beta_total_late10"],
                "beta_hard_full": row["beta_hard_full"],
                "beta_soft_full": row["beta_soft_full"],
                "r2_min": min(
                    row["r2_total_early10"],
                    row["r2_total_late10"],
                    row["r2_hard_full"],
                    row["r2_soft_full"],
                ),
                "soft_gt_hard_gt_total_early": bool(
                    row["beta_soft_full"] > row["beta_hard_full"] > row["beta_total_early10"]
                ),
                "soft_gt_hard_gt_total_late": bool(
                    row["beta_soft_full"] > row["beta_hard_full"] > row["beta_total_late10"]
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "r2_1_slope_ordering_ladder_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), sharey=True)
    x = np.arange(4)
    xlabels = ["total\n0-10 ps", "total\nscreened", "hard\nstate", "soft\nstate"]
    temp_colors = {
        298: "#345f8c",
        353: "#4f8d64",
        373: "#b57635",
        423: "#9a3f4f",
    }
    for ax, anion in zip(axes, ["fsi", "tfsi", "beti"]):
        sub = slopes[slopes["anion"] == anion]
        for _, row in sub.iterrows():
            y = [
                row["beta_total_early10"],
                row["beta_total_late10"],
                row["beta_hard_full"],
                row["beta_soft_full"],
            ]
            ax.plot(
                x,
                y,
                marker="o",
                lw=1.5,
                ms=4,
                color=temp_colors.get(int(row["T"]), "#555555"),
                label=f"{int(row['T'])} K",
                alpha=0.92,
            )
        ax.text(
            0.05,
            0.96,
            anion.upper(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel("apparent finite-window survival slope" if anion == "fsi" else "")
        ax.set_ylim(0, max(1.4, float(slopes["beta_soft_full"].max()) * 1.08))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    savefig(fig, "r2_1_slope_ladder_twostate_justification", rect=(0.02, 0.02, 0.98, 0.90))
    return summary


def write_machine_summary(
    delta_summary: pd.DataFrame,
    exp_summary: pd.DataFrame,
    case_df: pd.DataFrame,
    slope_summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# R2-1 Intuitive AIC/BIC Numeric Summary")
    lines.append("")
    lines.append("Delta score rule used here: 0 is best, <=2 is competitive, and >10 is weak support relative to the best model.")
    lines.append("")
    lines.append("## Exponential rejection at tmin = 2 ps")
    t2 = exp_summary[exp_summary["tmin"] == 2.0].copy()
    for _, row in t2.sort_values(["dataset", "criterion"]).iterrows():
        lines.append(
            "- "
            f"{DATASET_LABELS[row['dataset']]}, {row['criterion']}: "
            f"median Delta={row['median_delta_exponential']:.2g}, "
            f"competitive {int(row['n_competitive_delta_le_2'])}/{int(row['n_cases'])}, "
            f"weak-support {int(row['n_unsupported_delta_gt_10'])}/{int(row['n_cases'])}."
        )
    lines.append("")
    lines.append("## Best-model counts at tmin = 2 ps")
    t2_cases = case_df[case_df["tmin"] == 2.0]
    for dataset in DATASET_ORDER:
        for criterion in ["AIC", "BIC"]:
            sub = t2_cases[(t2_cases["dataset"] == dataset) & (t2_cases["criterion"] == criterion)]
            counts = sub["best_model"].value_counts().reindex(MODEL_ORDER).dropna().astype(int)
            count_text = ", ".join(f"{clean_model_name(k)}={v}" for k, v in counts.items())
            lines.append(f"- {DATASET_LABELS[dataset]}, {criterion}: {count_text}.")
    lines.append("")
    lines.append("## Slope-ordering check")
    n = len(slope_summary)
    early_ok = int(slope_summary["soft_gt_hard_gt_total_early"].sum())
    late_ok = int(slope_summary["soft_gt_hard_gt_total_late"].sum())
    min_r2 = float(slope_summary["r2_min"].min())
    lines.append(
        f"- soft > hard > total early-window slope: {early_ok}/{n}; "
        f"soft > hard > total tail-screened slope: {late_ok}/{n}; minimum reported R2={min_r2:.3f}."
    )
    lines.append("")
    lines.append("## Generated figures")
    for stem in [
        "r2_1_aic_bic_delta_vs_tmin",
        "r2_1_exponential_rejection_aic_bic",
        "r2_1_best_model_counts_aic_bic",
        "r2_1_best_model_heatmap_total_interevent",
        "r2_1_best_model_heatmap_total_survival",
        "r2_1_best_model_heatmap_state_duration",
        "r2_1_best_model_heatmap_state_survival",
        "r2_1_slope_ladder_twostate_justification",
    ]:
        lines.append(f"- {FIG_DIR / (stem + '.pdf')}")
    lines.append("")
    (OUT_DIR / "r2_1_intuitive_bic_numeric_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_style()
    df = finite_fit_table()
    delta_summary = make_delta_summary(df)
    case_df, count_df = make_best_model_tables(df)
    plot_delta_vs_tmin(delta_summary)
    exp_summary = plot_exponential_rejection(df)
    plot_best_model_heatmaps(case_df)
    plot_best_model_counts(count_df)
    slope_summary = plot_slope_ladder()
    write_machine_summary(delta_summary, exp_summary, case_df, slope_summary)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote summary tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
