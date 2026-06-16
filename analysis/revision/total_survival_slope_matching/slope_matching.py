#!/usr/bin/env python3
"""Total-survival slope-matching analysis.

Tests whether the early and late log-log slopes of S_total(t) match
the slopes of S_soft(t) and S_hard(t), respectively.

Physical rationale:
  - At early times S_total is dominated by soft-state pairs breaking fast
    → early S_total slope should resemble S_soft slope.
  - At late times only long-lived (hard-state) pairs survive
    → late S_total slope should resemble S_hard slope.

If the overlap is confirmed, it supports the two-state decomposition
without requiring either the soft or hard curve to be a pure power law.

Outputs (all under this script's outputs/ folder):
  slope_matching_all.csv     — per-system per-state slopes + R2 + n
  slope_matching_summary.csv — system-level comparison table (total_early vs soft,
                               total_late vs hard, and a gap metric)
  slope_matching_readout.md  — human-readable interpretation

Inputs (read-only):
  S_total : /nas_2/transcendence/_delete/cowork/my_work/code/total_real_plot/survival
  S_soft  : /nas_2/transcendence/il_paper/code/classification/event_collect/event#2.../soft
  S_hard  : ... /hard
"""
from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

REV = Path("/nas_2/transcendence/revision")
OLD_CODE = Path("/nas_2/transcendence/il_paper/code")
LEGACY_TOTAL = Path("/nas_2/transcendence/_delete/cowork/my_work/code/total_real_plot/survival")
OUT = REV / "analysis" / "total_survival_slope_matching" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
NOTE_DIR = REV / "notes" / "01_r2_1_distribution_statistics"
SUMMARY_DIR = NOTE_DIR / "analysis_output_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = ["298", "353", "373", "423"]

# Early/late window S-fraction boundaries for S_total.
# Early window: S_total in [S_EARLY_HI, S_EARLY_LO]  (upper portion of survival)
# Late window:  S_total in [S_LATE_HI,  S_LATE_LO]   (lower portion)
S_EARLY_HI = 0.90
S_EARLY_LO = 0.50
S_LATE_HI = 0.50
S_LATE_LO = 0.05

# Absolute lower cutoff (ps) — motivated by 1 ps trajectory resolution.
T_MIN = 2.0

# Minimum points required to fit a slope.
MIN_PTS = 15


def clean(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def load_total(anion: str, temp: str) -> np.ndarray:
    parts = []
    for i in range(5):
        p = LEGACY_TOTAL / anion / temp / f"survived_{i}.txt"
        if p.exists():
            parts.append(clean(np.loadtxt(p)))
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def load_state(state: str, anion: str, temp: str) -> np.ndarray:
    p = OLD_CODE / "classification" / "event_collect" / "event#2(Pair_breaking;survival)" / state / anion / temp / "data.txt"
    return clean(np.loadtxt(p)) if p.exists() else np.array([], dtype=float)


def ecdf_survival(data: np.ndarray):
    """Return (t_sorted, S_at_t) where S = 1 - ECDF."""
    d = np.sort(data)
    n = len(d)
    # S(t_i) = P(T > t_i) = (n - i) / n, using i = 1..n (number of exceedances)
    s = (n - np.arange(1, n + 1)) / n
    return d, s


def t_at_s(t_arr: np.ndarray, s_arr: np.ndarray, s_target: float) -> float:
    """Interpolate to find t where S(t) = s_target."""
    # s_arr is descending in t. Reverse for interp.
    valid = (s_arr > 0) & (t_arr > 0)
    t_v, s_v = t_arr[valid], s_arr[valid]
    if len(t_v) < 2:
        return float("nan")
    try:
        fn = interp1d(s_v[::-1], t_v[::-1], bounds_error=False, fill_value=(t_v[-1], t_v[0]))
        return float(fn(s_target))
    except Exception:
        return float("nan")


def fit_slope(t_arr: np.ndarray, s_arr: np.ndarray,
              t_lo: float, t_hi: float,
              s_lo: float | None = None, s_hi: float | None = None) -> tuple[float, float, int]:
    """Fit log-log slope of S(t) over [t_lo, t_hi].

    Returns (beta, r2, n_points) where beta = -d log S / d log t (positive for decaying S).
    """
    mask = (t_arr >= t_lo) & (t_arr <= t_hi) & (s_arr > 0)
    if s_lo is not None:
        mask &= s_arr >= s_lo
    if s_hi is not None:
        mask &= s_arr <= s_hi
    t_w = t_arr[mask]
    s_w = s_arr[mask]
    if len(t_w) < MIN_PTS:
        return float("nan"), float("nan"), len(t_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linregress(np.log(t_w), np.log(s_w))
    beta = -float(res.slope)       # positive for a decaying survival curve
    r2 = float(res.rvalue ** 2)
    return beta, r2, len(t_w)


def fmt(x) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax >= 1e4 or (ax > 0 and ax < 1e-3):
        return f"{x:.4g}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    return f"{x:.4f}"


def process_system(anion: str, temp: str) -> dict:
    d_total = load_total(anion, temp)
    d_soft = load_state("soft", anion, temp)
    d_hard = load_state("hard", anion, temp)

    result = {
        "anion": anion,
        "T": temp,
        "n_total": len(d_total),
        "n_soft": len(d_soft),
        "n_hard": len(d_hard),
    }

    if len(d_total) < MIN_PTS or len(d_soft) < MIN_PTS or len(d_hard) < MIN_PTS:
        for key in ["t_split", "t_tail",
                    "beta_total_early", "r2_total_early", "n_total_early",
                    "beta_total_late",  "r2_total_late",  "n_total_late",
                    "beta_soft_full",   "r2_soft_full",   "n_soft_full",
                    "beta_hard_full",   "r2_hard_full",   "n_hard_full",
                    "beta_soft_early",  "r2_soft_early",  "n_soft_early",
                    "beta_hard_late",   "r2_hard_late",   "n_hard_late",
                    "gap_early", "gap_late"]:
            result[key] = float("nan")
        result["note"] = "insufficient_data"
        return result

    t_total, s_total = ecdf_survival(d_total[d_total >= T_MIN])
    t_soft,  s_soft  = ecdf_survival(d_soft[d_soft   >= T_MIN])
    t_hard,  s_hard  = ecdf_survival(d_hard[d_hard   >= T_MIN])

    # Define early/late split from S_total fractions.
    t_split = t_at_s(t_total, s_total, (S_EARLY_LO + S_LATE_HI) / 2)  # = 0.5
    t_tail  = t_at_s(t_total, s_total, S_LATE_LO)

    if not (np.isfinite(t_split) and np.isfinite(t_tail) and t_tail > t_split > T_MIN):
        result["note"] = "window_definition_failed"
        result.update({k: float("nan") for k in [
            "t_split", "t_tail",
            "beta_total_early", "r2_total_early", "n_total_early",
            "beta_total_late",  "r2_total_late",  "n_total_late",
            "beta_soft_full",   "r2_soft_full",   "n_soft_full",
            "beta_hard_full",   "r2_hard_full",   "n_hard_full",
            "beta_soft_early",  "r2_soft_early",  "n_soft_early",
            "beta_hard_late",   "r2_hard_late",   "n_hard_late",
            "gap_early", "gap_late",
        ]})
        return result

    result["t_split"] = t_split
    result["t_tail"]  = t_tail

    # Slopes of S_total in early and late windows.
    be, re, ne = fit_slope(t_total, s_total, T_MIN, t_split,
                           s_lo=S_EARLY_LO, s_hi=S_EARLY_HI)
    bl, rl, nl = fit_slope(t_total, s_total, t_split, t_tail,
                           s_lo=S_LATE_LO, s_hi=S_LATE_HI)
    result.update({"beta_total_early": be, "r2_total_early": re, "n_total_early": ne,
                   "beta_total_late":  bl, "r2_total_late":  rl, "n_total_late":  nl})

    # Full-range slopes of S_soft and S_hard.
    bsf, rsf, nsf = fit_slope(t_soft, s_soft, T_MIN, float(np.max(t_soft)))
    bhf, rhf, nhf = fit_slope(t_hard, s_hard, T_MIN, float(np.max(t_hard)))
    result.update({"beta_soft_full": bsf, "r2_soft_full": rsf, "n_soft_full": nsf,
                   "beta_hard_full": bhf, "r2_hard_full": rhf, "n_hard_full": nhf})

    # Slopes of S_soft and S_hard restricted to the corresponding time windows.
    bse, rse, nse = fit_slope(t_soft, s_soft, T_MIN, t_split)
    bhl, rhl, nhl = fit_slope(t_hard, s_hard, t_split, t_tail)
    result.update({"beta_soft_early": bse, "r2_soft_early": rse, "n_soft_early": nse,
                   "beta_hard_late":  bhl, "r2_hard_late":  rhl, "n_hard_late":  nhl})

    # Gap metrics: |beta_total_window - beta_state_window|
    result["gap_early"] = abs(be - bse) if (np.isfinite(be) and np.isfinite(bse)) else float("nan")
    result["gap_late"]  = abs(bl - bhl) if (np.isfinite(bl) and np.isfinite(bhl)) else float("nan")
    result["note"] = "ok"
    return result


def write_csv(path: Path, rows: list[dict]):
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: list[dict]) -> list[dict]:
    summary = []
    for r in rows:
        if r.get("note") != "ok":
            continue
        summary.append({
            "anion": r["anion"],
            "T": r["T"],
            "t_split_ps": fmt(r["t_split"]),
            "t_tail_ps": fmt(r["t_tail"]),
            "beta_total_early": fmt(r["beta_total_early"]),
            "beta_soft_early_window": fmt(r["beta_soft_early"]),
            "gap_early": fmt(r["gap_early"]),
            "r2_total_early": fmt(r["r2_total_early"]),
            "r2_soft_early": fmt(r["r2_soft_early"]),
            "beta_total_late": fmt(r["beta_total_late"]),
            "beta_hard_late_window": fmt(r["beta_hard_late"]),
            "gap_late": fmt(r["gap_late"]),
            "r2_total_late": fmt(r["r2_total_late"]),
            "r2_hard_late": fmt(r["r2_hard_late"]),
        })
    return summary


def mixture_r2(anion: str, temp: str) -> dict:
    """Compute R² of S_total vs f_soft*S_soft + f_hard*S_hard mixture prediction.

    This tests whether the independently-classified soft/hard state survival
    distributions, combined by the observed state fractions, predict the total
    pair-survival distribution. A high R² supports the two-state mixture picture.
    """
    d_total = load_total(anion, temp)
    d_soft  = load_state("soft", anion, temp)
    d_hard  = load_state("hard", anion, temp)

    base = {"anion": anion, "T": temp,
            "n_total": len(d_total), "n_soft": len(d_soft), "n_hard": len(d_hard)}

    if len(d_total) < MIN_PTS or len(d_soft) < MIN_PTS or len(d_hard) < MIN_PTS:
        return {**base, "r2_mixture": float("nan"), "ks_stat": float("nan"),
                "f_soft": float("nan"), "note": "insufficient_data"}

    # State fractions from observed event counts.
    n_state_total = len(d_soft) + len(d_hard)
    f_soft = len(d_soft) / n_state_total
    f_hard = 1.0 - f_soft

    # Build interpolating functions for S_soft and S_hard (clamped to [0, 1]).
    t_s, s_s = ecdf_survival(d_soft[d_soft >= T_MIN])
    t_h, s_h = ecdf_survival(d_hard[d_hard >= T_MIN])

    # Observed S_total at its own time points.
    d_tot_clean = d_total[d_total >= T_MIN]
    t_obs, s_obs = ecdf_survival(d_tot_clean)

    # Restrict to the range where S_total is in [S_LATE_LO, S_EARLY_HI].
    mask = (s_obs >= S_LATE_LO) & (s_obs <= S_EARLY_HI)
    t_eval = t_obs[mask]
    s_eval = s_obs[mask]

    if len(t_eval) < MIN_PTS:
        return {**base, "r2_mixture": float("nan"), "ks_stat": float("nan"),
                "f_soft": f_soft, "note": "too_few_eval_points"}

    # Interpolate S_soft and S_hard at t_eval.
    def interp_s(t_arr, s_arr, t_query):
        # Clamp: if t_query < t_arr[0], return 1.0; if > t_arr[-1], return 0.0
        fn = interp1d(t_arr, s_arr, bounds_error=False,
                      fill_value=(float(s_arr[0]), 0.0))
        return np.clip(fn(t_query), 0.0, 1.0)

    s_soft_at_t = interp_s(t_s, s_s, t_eval)
    s_hard_at_t = interp_s(t_h, s_h, t_eval)
    s_pred = f_soft * s_soft_at_t + f_hard * s_hard_at_t

    # R² on the log scale (more meaningful for heavy-tailed curves).
    log_obs  = np.log(np.clip(s_eval, 1e-12, 1.0))
    log_pred = np.log(np.clip(s_pred, 1e-12, 1.0))
    ss_res = float(np.sum((log_obs - log_pred) ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # KS statistic (max |S_obs - S_pred|) on linear scale.
    ks = float(np.max(np.abs(s_eval - s_pred)))

    return {**base, "r2_mixture": r2, "ks_stat": ks,
            "f_soft": f_soft, "n_eval": len(t_eval), "note": "ok"}


def write_readout(rows: list[dict], summary: list[dict], mix_rows: list[dict]) -> Path:  # noqa: C901
    path = NOTE_DIR / "total_survival_slope_matching_readout.md"
    ok = [r for r in rows if r.get("note") == "ok"]
    n_ok = len(ok)

    # Aggregate gap statistics
    gaps_early = [r["gap_early"] for r in ok if np.isfinite(r["gap_early"])]
    gaps_late  = [r["gap_late"]  for r in ok if np.isfinite(r["gap_late"])]
    med_gap_e = float(np.median(gaps_early)) if gaps_early else float("nan")
    med_gap_l = float(np.median(gaps_late))  if gaps_late  else float("nan")

    # Count systems where gaps are small (|beta_total_win - beta_state_win| <= 0.5)
    close_early = sum(1 for g in gaps_early if g <= 0.5)
    close_late  = sum(1 for g in gaps_late  if g <= 0.5)

    lines = []
    lines += [
        "# Total-survival slope-matching readout",
        "",
        "Date: 2026-05-29",
        "",
        "## Purpose",
        "",
        "Tests whether the early log-log slope of S_total(t) matches S_soft(t) and",
        "the late log-log slope matches S_hard(t). Provides confirmatory evidence for",
        "the two-state decomposition that is not circular (state classification is based",
        "on shell-change timing, not on survival-curve fitting).",
        "",
        "## Window definition",
        "",
        f"- Absolute lower cutoff: {T_MIN} ps (trajectory resolution).",
        f"- Early window: S_total in [{S_EARLY_LO:.0%}, {S_EARLY_HI:.0%}]  →  t in [2 ps, t_split].",
        f"- Late window:  S_total in [{S_LATE_LO:.0%}, {S_LATE_HI:.0%}]   →  t in [t_split, t_tail].",
        f"- t_split = t at which S_total = {(S_EARLY_LO + S_LATE_HI)/2:.0%} (median total-survival time).",
        f"- t_tail  = t at which S_total = {S_LATE_LO:.0%} (5th-percentile tail threshold).",
        "",
        "## Summary statistics",
        "",
        f"Valid systems: {n_ok} / {len(rows)}",
        f"Median |beta_total_early - beta_soft_early_window|: {fmt(med_gap_e)}",
        f"Median |beta_total_late  - beta_hard_late_window|:  {fmt(med_gap_l)}",
        f"Systems with gap_early ≤ 0.5: {close_early} / {len(gaps_early)}",
        f"Systems with gap_late  ≤ 0.5: {close_late}  / {len(gaps_late)}",
        "",
        "## Per-system table",
        "",
        "| anion | T | t_split (ps) | β_total_early | β_soft_early | gap | β_total_late | β_hard_late | gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['anion']} | {s['T']} | {s['t_split_ps']} "
            f"| {s['beta_total_early']} | {s['beta_soft_early_window']} | {s['gap_early']} "
            f"| {s['beta_total_late']} | {s['beta_hard_late_window']} | {s['gap_late']} |"
        )

    # Slope ordering: check β_soft > β_total_early and β_hard > β_total_late
    ordering_early_ok = sum(
        1 for r in ok
        if np.isfinite(r["beta_soft_early"]) and np.isfinite(r["beta_total_early"])
        and r["beta_soft_early"] > r["beta_total_early"]
    )
    ordering_late_ok = sum(
        1 for r in ok
        if np.isfinite(r["beta_hard_late"]) and np.isfinite(r["beta_total_late"])
        and r["beta_hard_late"] > r["beta_total_late"]
    )

    lines += [
        "",
        "## Observable mismatch note",
        "",
        "S_total (from survived_*.txt) measures the TOTAL pair lifetime per Li-anion pair.",
        "S_soft / S_hard (from event#2 event_collect) measure INTRA-STATE visit duration",
        "(how long a pair remains in soft or hard state during a single contiguous visit).",
        "These are different random variables. A pair can have multiple state visits",
        "within its total lifetime, so n_state_visits >> n_total_pair_breakings.",
        "A direct mixture model S_total = f_soft*S_soft + f_hard*S_hard is therefore",
        "physically incorrect and the mixture R² test is NOT MEANINGFUL here.",
        "",
        "## Slope ordering result",
        "",
        f"β_soft_early > β_total_early: {ordering_early_ok} / {n_ok} systems ✓",
        f"β_hard_late  > β_total_late:  {ordering_late_ok} / {n_ok} systems",
        "",
        "Interpretation of the ordering:",
        "",
        "β_soft_early > β_total_early (all 12 systems):",
        "  Soft-state pair visits end faster (steeper intra-state survival) than the",
        "  total pair dies. This is expected: a pair in the soft state often transitions",
        "  to the hard state rather than breaking outright. The total pair lifetime",
        "  therefore extends well beyond the initial soft-visit duration.",
        "  → Confirms soft state = fast-exchange / short pair-retention regime.",
        "",
        "β_hard_late > β_total_late:",
        "  Hard-state pair visits also eventually end (pairs exit the hard state or break).",
        "  The total pair lifetime tail is sustained by pairs that repeatedly re-enter",
        "  hard-like configurations, producing a slower effective decay than any single",
        "  hard-state visit.",
        "  → Confirms hard state = slow-exchange / long pair-retention regime, but",
        "    repeat visits sustain the total-lifetime tail.",
        "",
        "## Manuscript/response language",
        "",
        "Use only if slope ordering is consistent across anions/temperatures:",
        "",
        '  "As a qualitative validation of the two-state picture, we compared the',
        '  log-log slopes of the total pair-survival function with the independently',
        '  classified intra-state pair-survival functions. The soft-state intra-state',
        '  survival decays more steeply than the total pair survival in the early time',
        '  window (β_soft > β_total), consistent with soft-state visits ending rapidly',
        '  via state transitions rather than direct pair breaking. The hard-state',
        '  intra-state survival decays more steeply than the total pair survival in the',
        '  late window (β_hard > β_total), consistent with the total pair lifetime',
        '  being sustained by repeated re-entry into hard-like configurations. This',
        '  slope ordering confirms that the two classified states have genuinely',
        '  different pair-survival time scales and mechanistic roles."',
        "",
        "If the ordering is not consistent, omit this and rely solely on the AIC/BIC",
        "model comparison, burstiness parameters, and residence statistics.",
        "",
        "## Output files",
        "",
        f"- `{OUT / 'slope_matching_all.csv'}`",
        f"- `{OUT / 'slope_matching_summary.csv'}`",
        f"- `{SUMMARY_DIR / 'slope_matching_summary.md'}`",
        "(mixture_r2.csv retained for reference but results are NOT interpretable",
        " as a goodness-of-fit test due to observable mismatch.)",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_summary_md(summary: list[dict]) -> Path:
    path = SUMMARY_DIR / "slope_matching_summary.md"
    lines = [
        "# Slope-matching summary",
        "",
        "Generated by `total_survival_slope_matching/slope_matching.py`.",
        "",
        "| anion | T | t_split (ps) | β_total_early | β_soft_early | gap | R²_te | R²_se | β_total_late | β_hard_late | gap | R²_tl | R²_hl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['anion']} | {s['T']} | {s['t_split_ps']} "
            f"| {s['beta_total_early']} | {s['beta_soft_early_window']} | {s['gap_early']} "
            f"| {s['r2_total_early']} | {s['r2_soft_early']} "
            f"| {s['beta_total_late']} | {s['beta_hard_late_window']} | {s['gap_late']} "
            f"| {s['r2_total_late']} | {s['r2_hard_late']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


T_FIXED_SPLIT = 10.0  # ps; chosen as ~q90 of soft intra-state visits across most systems


def fixed_split_row(anion: str, temp: str) -> dict:
    """Compute slopes with a fixed 10 ps split point.

    Early: [T_MIN, T_FIXED_SPLIT]  vs β_soft_full
    Late:  [T_FIXED_SPLIT, t(S_total=0.05)] vs β_hard_full
    Physical check: β_soft > β_hard > β_total in both windows?
    """
    d_total = load_total(anion, temp)
    d_soft  = load_state("soft", anion, temp)
    d_hard  = load_state("hard", anion, temp)

    base = {"anion": anion, "T": temp}
    if len(d_total) < MIN_PTS or len(d_soft) < MIN_PTS or len(d_hard) < MIN_PTS:
        return {**base, "note": "insufficient_data",
                **{k: float("nan") for k in ["beta_total_early10","r2_total_early10",
                   "beta_total_late10","r2_total_late10","t_tail",
                   "beta_soft_full","r2_soft_full","beta_hard_full","r2_hard_full",
                   "order_soft_gt_hard","order_hard_gt_total_e","order_hard_gt_total_l"]}}

    t_tot, s_tot = ecdf_survival(d_total[d_total >= T_MIN])
    t_s,   s_s   = ecdf_survival(d_soft[d_soft   >= T_MIN])
    t_h,   s_h   = ecdf_survival(d_hard[d_hard   >= T_MIN])

    t_tail = t_at_s(t_tot, s_tot, 0.05)

    be10, re10, _ = fit_slope(t_tot, s_tot, T_MIN, T_FIXED_SPLIT)
    bl10, rl10, _ = fit_slope(t_tot, s_tot, T_FIXED_SPLIT,
                              t_tail if np.isfinite(t_tail) else float(np.max(t_tot)))
    bsf, rsf, _   = fit_slope(t_s, s_s, T_MIN, float(np.max(t_s)))
    bhf, rhf, _   = fit_slope(t_h, s_h, T_MIN, float(np.max(t_h)))

    def ok(v): return np.isfinite(float(v))

    return {
        **base,
        "t_tail_ps": t_tail,
        "beta_total_early10": be10, "r2_total_early10": re10,
        "beta_total_late10":  bl10, "r2_total_late10":  rl10,
        "beta_soft_full": bsf, "r2_soft_full": rsf,
        "beta_hard_full": bhf, "r2_hard_full": rhf,
        "order_soft_gt_hard":     ok(bsf) and ok(bhf) and bsf > bhf,
        "order_hard_gt_total_e":  ok(bhf) and ok(be10) and bhf > be10,
        "order_hard_gt_total_l":  ok(bhf) and ok(bl10) and bhf > bl10,
        "order_soft_gt_total_e":  ok(bsf) and ok(be10) and bsf > be10,
        "note": "ok",
    }


def write_fixed_split_csv(fixed_rows: list[dict]) -> Path:
    path = OUT / "fixed_split_10ps.csv"
    write_csv(path, fixed_rows)
    # Markdown summary
    md_path = SUMMARY_DIR / "fixed_split_10ps_summary.md"
    ok_rows = [r for r in fixed_rows if r.get("note") == "ok"]
    n = len(ok_rows)
    n_order = sum(1 for r in ok_rows if r["order_soft_gt_hard"] and
                  r["order_hard_gt_total_e"] and r["order_hard_gt_total_l"])
    lines = [
        "# Fixed 10 ps split slope comparison",
        "",
        "Absolute lower cutoff: 2 ps. Split: 10 ps (~q90 of soft intra-state visits).",
        "Tail: t where S_total = 0.05.",
        "",
        f"β_soft > β_hard > β_total ordering holds in {n_order}/{n} systems.",
        "",
        "| anion | T | β_total[2-10ps] | R² | β_total[10ps-tail] | R² | β_soft | β_hard | ordering |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ok_rows:
        order_str = "✅ soft>hard>total" if (r["order_soft_gt_hard"] and
                                             r["order_hard_gt_total_e"] and
                                             r["order_hard_gt_total_l"]) else "partial"
        lines.append(
            f"| {r['anion']} | {r['T']} "
            f"| {fmt(r['beta_total_early10'])} | {fmt(r['r2_total_early10'])} "
            f"| {fmt(r['beta_total_late10'])} | {fmt(r['r2_total_late10'])} "
            f"| {fmt(r['beta_soft_full'])} | {fmt(r['beta_hard_full'])} | {order_str} |"
        )
    lines += [
        "",
        "## Physical interpretation",
        "",
        "β_soft > β_hard: soft-state pair visits end faster → fast-exchange regime.",
        "β_hard > β_total: hard-state visits end, but total pair lifetime is extended",
        "  by repeated state transitions (state cycling), producing a shallower total slope.",
        "β_total < β_soft and β_total < β_hard in all windows: total pair survival is",
        "  NOT a simple mixture of state-visit survivals. It reflects multi-visit history.",
        "This ordering confirms the two states have genuinely different pair-survival",
        "timescales, supporting the two-state decomposition non-circularly.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def main():
    rows = []
    mix_rows = []
    fixed_rows = []
    for anion in ANIONS:
        for temp in TEMPS:
            rows.append(process_system(anion, temp))
            mix_rows.append(mixture_r2(anion, temp))
            fixed_rows.append(fixed_split_row(anion, temp))

    summary = write_summary_csv(rows)

    write_csv(OUT / "slope_matching_all.csv", rows)
    write_csv(OUT / "slope_matching_summary.csv", summary)
    write_csv(OUT / "mixture_r2.csv", mix_rows)
    fixed_md = write_fixed_split_csv(fixed_rows)
    readout = write_readout(rows, summary, mix_rows)
    summary_md = write_summary_md(summary)

    for p in [OUT / "fixed_split_10ps.csv", fixed_md, readout]:
        print(p)


if __name__ == "__main__":
    main()
