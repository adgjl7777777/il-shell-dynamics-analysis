#!/usr/bin/env python3
"""Sensitivity audit for shifted power-law handling in state survival fits.

This script compares the revision likelihood models with additional
shifted-power candidates and with the legacy log-binned regression convention.
It is intentionally limited to soft/hard intra-state pair-survival durations,
because that is where the old hard-state onset offset and shifted x-axis
treatment matter most.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import importlib.util
import math
import sys
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import linregress


REV = Path("/nas_2/transcendence/revision")
OLD_CODE = Path("/nas_2/transcendence/il_paper/code")
COMPARE_PATH = REV / "analysis" / "powerlaw_model_comparison" / "compare_models.py"
OUT = REV / "analysis" / "shifted_power_sensitivity" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
NOTE_DIR = REV / "notes" / "01_r2_1_distribution_statistics"
SUMMARY_DIR = NOTE_DIR / "analysis_output_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = ["298", "353", "373", "423"]
TRAIL = "1.0"
MIN_N = 30
EPS = 1e-300

TABLE2_NSTART = {
    "fsi": {"298": 33.3, "353": 20.0, "373": 17.7, "423": 13.5},
    "tfsi": {"298": 159.4, "353": 80.9, "373": 59.1, "423": 36.9},
    "beti": {"298": 178.6, "353": 93.4, "373": 68.8, "423": 43.9},
}


spec = importlib.util.spec_from_file_location("compare_models", COMPARE_PATH)
cm = importlib.util.module_from_spec(spec)
sys.modules["compare_models"] = cm
assert spec.loader is not None
spec.loader.exec_module(cm)


@dataclass
class Fit:
    model: str
    ok: bool
    n_params: int
    loglik: float
    aic: float
    bic: float
    params: dict
    note: str = ""


def info_criteria(loglik: float, k: int, n: int) -> tuple[float, float]:
    return 2 * k - 2 * loglik, k * math.log(max(n, 1)) - 2 * loglik


def clean(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def load_current_event_collect(state: str, anion: str, temp: str) -> np.ndarray:
    path = (
        OLD_CODE
        / "classification"
        / "event_collect"
        / "event#2(Pair_breaking;survival)"
        / state
        / anion
        / temp
        / "data.txt"
    )
    try:
        return clean(np.loadtxt(path))
    except Exception:
        return np.array([], dtype=float)


def load_loyal_intervals(cat: str, anion: str, temp: str) -> np.ndarray:
    out = []
    base = OLD_CODE / "classification" / "x" / anion / temp / cat
    for i in range(5):
        path = base / f"{TRAIL}_{i}.txt"
        try:
            arr = np.loadtxt(path)
        except Exception:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size == 2:
                arr = arr.reshape(1, 2)
            else:
                continue
        if arr.shape[-1] < 2:
            continue
        for row in arr.reshape(-1, arr.shape[-1]):
            dur = float(row[1] - row[0])
            if np.isfinite(dur) and dur > 0:
                out.append(dur)
    return np.asarray(out, dtype=float)


def candidate_datasets():
    for anion in ANIONS:
        for temp in TEMPS:
            soft_current = load_current_event_collect("soft", anion, temp)
            hard_offset_current = load_current_event_collect("hard", anion, temp)
            soft_raw = load_loyal_intervals("x11_loyal", anion, temp)
            hard_raw = load_loyal_intervals("x22_loyal", anion, temp)
            nstart = TABLE2_NSTART[anion][temp]
            hard_reoffset = clean(hard_raw - nstart)
            yield {
                "input_variant": "soft_current_event_collect",
                "state": "soft",
                "anion": anion,
                "T": temp,
                "nstart": 0.0,
                "data": soft_current,
                "note": "soft event_collect data; no n_start subtraction",
            }
            yield {
                "input_variant": "soft_raw_x11_loyal",
                "state": "soft",
                "anion": anion,
                "T": temp,
                "nstart": 0.0,
                "data": soft_raw,
                "note": "raw x11_loyal interval durations reconstructed from classification/x",
            }
            yield {
                "input_variant": "hard_current_event_collect_offset",
                "state": "hard",
                "anion": anion,
                "T": temp,
                "nstart": nstart,
                "data": hard_offset_current,
                "note": "hard event_collect data; already duration - n_start",
            }
            yield {
                "input_variant": "hard_raw_x22_loyal_no_offset",
                "state": "hard",
                "anion": anion,
                "T": temp,
                "nstart": 0.0,
                "data": hard_raw,
                "note": "raw x22_loyal interval durations; no n_start subtraction",
            }
            yield {
                "input_variant": "hard_raw_x22_loyal_minus_nstart",
                "state": "hard",
                "anion": anion,
                "T": temp,
                "nstart": nstart,
                "data": hard_reoffset,
                "note": "raw x22_loyal interval durations minus TABLE2_NSTART",
            }


def fit_shifted_pareto_min(x: np.ndarray) -> Fit:
    if len(x) < MIN_N:
        return Fit("shifted_pareto_min", False, 2, -math.inf, math.inf, math.inf, {}, "n<MIN_N")
    x0 = float(np.min(x) - 1.0)
    y = x - x0
    y = y[y >= 1.0]
    denom = float(np.sum(np.log(y)))
    if denom <= 0:
        return Fit("shifted_pareto_min", False, 2, -math.inf, math.inf, math.inf, {}, "nonpositive_log_denominator")
    alpha = 1.0 + len(y) / denom
    ll = float(np.sum(math.log(alpha - 1.0) - alpha * np.log(y)))
    # Penalize x0 as an effective extra parameter because it is chosen from data.
    aic, bic = info_criteria(ll, 2, len(y))
    return Fit("shifted_pareto_min", True, 2, ll, aic, bic, {"alpha": alpha, "x0": x0, "ymin": 1.0})


def fit_shifted_tempered_min(x: np.ndarray) -> Fit:
    if len(x) < MIN_N:
        return Fit("shifted_tempered_min", False, 3, -math.inf, math.inf, math.inf, {}, "n<MIN_N")
    x0 = float(np.min(x) - 1.0)
    y = x - x0
    y = y[y >= 1.0]

    cache = {}

    def log_norm(alpha: float, lam: float) -> float:
        key = (round(alpha, 10), round(lam, 14))
        if key in cache:
            return cache[key]

        def integrand(t):
            return (t ** (-alpha)) * math.exp(-lam * t)

        try:
            z, _ = quad(integrand, 1.0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        except Exception:
            z = 0.0
        out = math.log(z) if np.isfinite(z) and z > 0 else math.inf
        cache[key] = out
        return out

    def nll(theta):
        alpha, log_lam = theta
        lam = math.exp(log_lam)
        if alpha <= 0 or alpha > 10 or lam <= 0 or not np.isfinite(lam):
            return 1e300
        lz = log_norm(alpha, lam)
        if not np.isfinite(lz):
            return 1e300
        ll = -alpha * np.sum(np.log(y)) - lam * np.sum(y) - len(y) * lz
        val = -float(ll)
        return val if np.isfinite(val) else 1e300

    mean_y = max(float(np.mean(y)), 1.0)
    starts = []
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for lam in [1.0 / mean_y, 0.1 / mean_y, 1.0 / max(float(np.max(y)), mean_y)]:
            starts.append((alpha, math.log(max(lam, 1e-12))))

    best = None
    bounds = [(0.01, 10.0), (-30.0, 5.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for st in starts:
            res = minimize(nll, np.array(st), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
            if best is None or res.fun < best.fun:
                best = res
    if best is None or not np.isfinite(best.fun):
        return Fit("shifted_tempered_min", False, 3, -math.inf, math.inf, math.inf, {}, "opt_failed")
    alpha = float(best.x[0])
    lam = math.exp(float(best.x[1]))
    ll = -float(best.fun)
    # Penalize x0 as an effective extra parameter because it is chosen from data.
    aic, bic = info_criteria(ll, 3, len(y))
    return Fit("shifted_tempered_min", True, 3, ll, aic, bic, {"alpha": alpha, "lambda": lam, "x0": x0, "ymin": 1.0})


IQRDIV = 2
BINCROSS = 10
DELTHRES = 1e-8


def get_distr(xs):
    distr = {}
    for x in xs:
        distr[x] = distr.setdefault(x, 0) + 1
    return distr


def get_logbin_mix(distr, binsize, binstart, bincross):
    distr_bin = {}
    bin_mix = {}
    b0 = binstart
    b1 = b0 + binsize
    bc = b0 + binsize * 0.5
    for x in sorted(distr.keys()):
        while math.log(x) >= b1:
            b0 += binsize
            b1 = b0 + binsize
            bc = b0 + binsize * 0.5
        xc = math.exp(bc)
        distr_bin[xc] = distr_bin.setdefault(xc, 0) + distr[x]
        if x < bincross:
            bin_mix[xc] = bin_mix.setdefault(xc, 0) + 1
        else:
            bin_mix[xc] = math.exp(b1) - math.exp(b0)
    return distr_bin, bin_mix


def legacy_pow_calc(real: np.ndarray) -> dict:
    real = clean(real)
    if len(real) <= 1:
        return {"ok": False, "note": "n<=1"}
    q25, q75 = np.log(np.quantile(real, 0.25)), np.log(np.quantile(real, 0.75))
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        return {"ok": False, "note": "nonpositive_log_iqr"}
    binsize = 2 * iqr / (len(real) ** (1 / 3)) / IQRDIV
    binstart = math.log(float(np.min(real)))
    distr_bin, bin_mix = get_logbin_mix(get_distr(real), binsize, binstart, BINCROSS)
    for x in list(distr_bin):
        distr_bin[x] /= bin_mix[x]
    xs = sorted(distr_bin.keys())
    ys = np.asarray([distr_bin[i] for i in xs], dtype=float)
    ys = ys / ys.sum()
    fx, fy = [], []
    for xi, yi in zip(xs, ys):
        if yi > DELTHRES and xi > DELTHRES:
            fx.append(xi - xs[0] + 1.0)
            fy.append(yi)
    if len(fx) <= 1:
        return {"ok": False, "note": "n_fit<=1", "first_bin": xs[0] if xs else math.nan}
    logx = np.log(np.asarray(fx))
    logy = np.log(np.asarray(fy))
    result = linregress(logx, logy)
    return {
        "ok": True,
        "beta": -float(result.slope),
        "beta_se": float(result.stderr) if result.stderr is not None else math.nan,
        "intercept": float(result.intercept),
        "intercept_se": float(result.intercept_stderr) if result.intercept_stderr is not None else math.nan,
        "r2": float(result.rvalue ** 2),
        "n_fit_bins": len(fx),
        "first_bin": float(xs[0]),
        "xmin_raw": float(np.min(real)),
        "xmax_raw": float(np.max(real)),
        "note": "legacy shifted log-binned PDF regression: x_fit = bin_center - first_bin + 1",
    }


def fit_all(data: np.ndarray) -> list[Fit]:
    x = clean(data)
    if len(x) < MIN_N:
        return [Fit("all_models", False, 0, -math.inf, math.inf, math.inf, {}, "n<MIN_N")]
    xmin = float(np.min(x))
    base = []
    for r in cm.fit_all_for_dataset(x, xmin):
        base.append(Fit(r.model, r.ok, r.n_params, r.loglik, r.aic, r.bic, dict(r.params), r.note))
    base.append(fit_shifted_pareto_min(x))
    base.append(fit_shifted_tempered_min(x))
    return base


def write_csv(path: Path, rows: list[dict]):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pack(counter: Counter) -> str:
    return "; ".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


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


def main():
    fit_rows = []
    best_rows = []
    legacy_rows = []
    inventory_rows = []

    for meta in candidate_datasets():
        data = clean(meta["data"])
        inventory_rows.append({
            "input_variant": meta["input_variant"],
            "state": meta["state"],
            "anion": meta["anion"],
            "T": meta["T"],
            "n": len(data),
            "nstart": meta["nstart"],
            "xmin": float(np.min(data)) if len(data) else math.nan,
            "xmax": float(np.max(data)) if len(data) else math.nan,
            "mean": float(np.mean(data)) if len(data) else math.nan,
            "median": float(np.median(data)) if len(data) else math.nan,
            "note": meta["note"],
        })
        legacy = legacy_pow_calc(data)
        legacy_rows.append({
            "input_variant": meta["input_variant"],
            "state": meta["state"],
            "anion": meta["anion"],
            "T": meta["T"],
            "n": len(data),
            **legacy,
        })
        fits = fit_all(data)
        valid = [r for r in fits if r.ok]
        best_aic = min((r.aic for r in valid), default=math.inf)
        best_bic = min((r.bic for r in valid), default=math.inf)
        by_model = {r.model: r for r in valid}
        for r in fits:
            fit_rows.append({
                "input_variant": meta["input_variant"],
                "state": meta["state"],
                "anion": meta["anion"],
                "T": meta["T"],
                "n": len(data),
                "model": r.model,
                "ok": r.ok,
                "n_params": r.n_params,
                "loglik": r.loglik,
                "AIC": r.aic,
                "BIC": r.bic,
                "delta_AIC": r.aic - best_aic if np.isfinite(best_aic) else math.nan,
                "delta_BIC": r.bic - best_bic if np.isfinite(best_bic) else math.nan,
                "params": ";".join(f"{k}={v:.8g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in r.params.items()),
                "note": r.note,
            })
        if valid:
            vals_aic = sorted(valid, key=lambda r: r.aic)
            vals_bic = sorted(valid, key=lambda r: r.bic)

            def delta(model):
                return by_model[model].aic - vals_aic[0].aic if model in by_model else math.nan

            best_rows.append({
                "input_variant": meta["input_variant"],
                "state": meta["state"],
                "anion": meta["anion"],
                "T": meta["T"],
                "n": len(data),
                "best_AIC_model": vals_aic[0].model,
                "best_BIC_model": vals_bic[0].model,
                "second_AIC_model": vals_aic[1].model if len(vals_aic) > 1 else "",
                "delta_AIC_second": vals_aic[1].aic - vals_aic[0].aic if len(vals_aic) > 1 else math.nan,
                "delta_AIC_unshifted_pareto": delta("pareto_power_law"),
                "delta_AIC_unshifted_tempered": delta("tempered_power_law"),
                "delta_AIC_shifted_pareto": delta("shifted_pareto_min"),
                "delta_AIC_shifted_tempered": delta("shifted_tempered_min"),
                "delta_AIC_biexponential": delta("biexponential_conditional"),
                "delta_AIC_weibull": delta("weibull_conditional"),
                "delta_AIC_exponential": delta("exponential_conditional"),
            })

    summary_rows = []
    for variant, vals in sorted(defaultdict(list, ((None, None),)).items()):
        pass

    grouped = defaultdict(list)
    for r in best_rows:
        grouped[r["input_variant"]].append(r)
    for variant, vals in sorted(grouped.items()):
        power_models = {"pareto_power_law", "tempered_power_law", "shifted_pareto_min", "shifted_tempered_min"}
        shifted_models = {"shifted_pareto_min", "shifted_tempered_min"}
        unshifted_models = {"pareto_power_law", "tempered_power_law"}
        summary_rows.append({
            "input_variant": variant,
            "cases": len(vals),
            "best_AIC_counts": pack(Counter(r["best_AIC_model"] for r in vals)),
            "best_BIC_counts": pack(Counter(r["best_BIC_model"] for r in vals)),
            "power_family_best_cases": sum(1 for r in vals if r["best_AIC_model"] in power_models),
            "shifted_power_best_cases": sum(1 for r in vals if r["best_AIC_model"] in shifted_models),
            "unshifted_power_best_cases": sum(1 for r in vals if r["best_AIC_model"] in unshifted_models),
            "median_delta_AIC_shifted_tempered": float(np.nanmedian([float(r["delta_AIC_shifted_tempered"]) for r in vals])),
            "median_delta_AIC_shifted_pareto": float(np.nanmedian([float(r["delta_AIC_shifted_pareto"]) for r in vals])),
            "median_delta_AIC_unshifted_tempered": float(np.nanmedian([float(r["delta_AIC_unshifted_tempered"]) for r in vals])),
            "median_delta_AIC_unshifted_pareto": float(np.nanmedian([float(r["delta_AIC_unshifted_pareto"]) for r in vals])),
        })

    write_csv(OUT / "shifted_power_inventory.csv", inventory_rows)
    write_csv(OUT / "shifted_power_model_fits_all.csv", fit_rows)
    write_csv(OUT / "shifted_power_best_models.csv", best_rows)
    write_csv(OUT / "legacy_binned_power_regression.csv", legacy_rows)
    write_csv(OUT / "shifted_power_count_summary.csv", summary_rows)

    md = []
    md.append("# Shifted power-law sensitivity")
    md.append("")
    md.append("This analysis asks whether the state-resolved survival conclusion changes when the old shifted power-law convention is made explicit.")
    md.append("")
    md.append("## What was tested")
    md.append("")
    md.append("- Current revision audit models: conditional exponential, Weibull/stretched exponential, unshifted Pareto power law, unshifted tempered power law, and biexponential mixture.")
    md.append("- Added shifted power-law candidates: `shifted_pareto_min` and `shifted_tempered_min`, where `y = x - x0` and `x0 = min(x) - 1`, so the fitted lower support is `y >= 1`.")
    md.append("- The shifted models are penalized by one additional effective parameter for `x0`, because the shift is chosen from the data.")
    md.append("- Legacy regression check: replicated the old log-binned PDF regression convention, where `x_fit = bin_center - first_bin + 1` before log-log linear regression.")
    md.append("- Hard-state inputs were tested both as raw `x22_loyal` durations and after subtracting `TABLE2_NSTART`.")
    md.append("")
    md.append("## Likelihood/AIC summary")
    md.append("")
    md.append("| Input variant | Cases | Best AIC counts | Shifted power best | Unshifted power best | Median delta AIC shifted tempered | Median delta AIC unshifted tempered |")
    md.append("|---|---:|---|---:|---:|---:|---:|")
    for r in summary_rows:
        md.append(f"| {r['input_variant']} | {r['cases']} | {r['best_AIC_counts']} | {r['shifted_power_best_cases']} | {r['unshifted_power_best_cases']} | {fmt(r['median_delta_AIC_shifted_tempered'])} | {fmt(r['median_delta_AIC_unshifted_tempered'])} |")
    md.append("")
    md.append("## Legacy shifted log-binned regression snapshot")
    md.append("")
    md.append("The legacy regression almost always returns a finite power-law slope because it performs a descriptive linear regression on the shifted, binned histogram. This is useful for reporting finite-window slopes, but it is not a likelihood-based proof that the full distribution is a pure power law.")
    md.append("")
    md.append("| Input variant | Median beta | Median R2 | Median fit bins |")
    md.append("|---|---:|---:|---:|")
    for variant in sorted(set(r["input_variant"] for r in legacy_rows)):
        vals = [r for r in legacy_rows if r["input_variant"] == variant and r.get("ok") is True]
        if not vals:
            continue
        md.append(f"| {variant} | {fmt(np.median([float(r['beta']) for r in vals]))} | {fmt(np.median([float(r['r2']) for r in vals]))} | {fmt(np.median([float(r['n_fit_bins']) for r in vals]))} |")
    md.append("")
    md.append("## Interpretation guide")
    md.append("")
    md.append("- If shifted power-law candidates win, then part of the old power-law appearance is tied to the onset-shift convention and should be described as a finite-window shifted-tail descriptor.")
    md.append("- If shifted power-law candidates still do not dominate, then the conclusion that state-resolved survival is composite/non-single-exponential is robust.")
    md.append("- The legacy shifted log-binned regression can remain useful for descriptive beta values, but reviewer-facing claims should rely on the likelihood/AIC/BIC comparison.")
    md.append("")
    md.append("## Output files")
    md.append("")
    for name in [
        "shifted_power_inventory.csv",
        "shifted_power_model_fits_all.csv",
        "shifted_power_best_models.csv",
        "legacy_binned_power_regression.csv",
        "shifted_power_count_summary.csv",
    ]:
        md.append(f"- `{OUT / name}`")
    (SUMMARY_DIR / "shifted_power_summary.md").write_text("\n".join(md) + "\n")

    note = []
    note.append("# Shifted power sensitivity readout")
    note.append("")
    note.append("Date: 2026-05-29")
    note.append("")
    note.append("Question: does the state-resolved survival result change if the old shifted power-law convention is made explicit?")
    note.append("")
    note.append("Short answer: see `shifted_power_summary.md` and `shifted_power_count_summary.csv`. Use this analysis to distinguish descriptive shifted log-log slopes from likelihood-supported distribution families.")
    note.append("")
    note.append("Main caution: the old `pow_calc` fit uses `x_fit = bin_center - first_bin + 1`, and the hard event-collect survival input is already `duration - n_start`. Therefore old beta values and revision likelihood results are not identical statistical objects.")
    note.append("")
    note.append("Recommended manuscript stance: keep beta as a finite-window descriptive slope; support burstiness with burstiness parameters and non-single-exponential model comparison; avoid saying the state-resolved survival distributions are pure power laws.")
    note.append("")
    note.append("Output summary:")
    for r in summary_rows:
        note.append(f"- {r['input_variant']}: {r['best_AIC_counts']}")
    (NOTE_DIR / "shifted_power_sensitivity_readout.md").write_text("\n".join(note) + "\n")

    print(SUMMARY_DIR / "shifted_power_summary.md")
    print(NOTE_DIR / "shifted_power_sensitivity_readout.md")


if __name__ == "__main__":
    main()
