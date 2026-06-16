#!/usr/bin/env python3
"""Likelihood/AIC/BIC model comparison for revision R2-1.

Reads existing old-project duration files and writes all outputs under the
revision folder. No old data, old code, or manuscript files are modified.

The goal is not to prove an asymptotic power law. The goal is to quantify
whether simple exponential relaxation is inadequate and whether power-law-like
or cutoff-tailed alternatives provide a better finite-window description.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import logsumexp

OLD_CODE = Path('/nas_2/transcendence/il_paper/code')
REV_DIR = Path('/nas_2/transcendence/revision')
OUT_DIR = REV_DIR / 'analysis' / 'powerlaw_model_comparison' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = REV_DIR / 'notes' / '01_r2_1_distribution_statistics' / 'analysis_output_summaries'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ANIONS = ['fsi', 'tfsi', 'beti']
TEMPS = ['298', '353', '373', '423']
WINDOWS = {
    'full': 0.0,
    'tail_q75': 0.75,
}
MIN_N = 30
EPS = 1e-300

@dataclass
class FitResult:
    model: str
    ok: bool
    n_params: int
    loglik: float
    aic: float
    bic: float
    params: dict
    note: str = ''


def positive_data(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def select_window(data: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    if q <= 0:
        xmin = float(np.min(data))
    else:
        xmin = float(np.quantile(data, q))
    subset = data[data >= xmin]
    return subset, xmin


def info_criteria(loglik: float, k: int, n: int) -> tuple[float, float]:
    return 2 * k - 2 * loglik, k * math.log(max(n, 1)) - 2 * loglik


def fail(model: str, k: int, note: str) -> FitResult:
    return FitResult(model, False, k, float('-inf'), float('inf'), float('inf'), {}, note)


def fit_exponential(x: np.ndarray, xmin: float) -> FitResult:
    y = x - xmin
    mean_y = float(np.mean(y))
    if mean_y <= 0:
        return fail('exponential_conditional', 1, 'zero_mean_after_xmin_shift')
    lam = 1.0 / mean_y
    ll = float(np.sum(np.log(lam) - lam * y))
    aic, bic = info_criteria(ll, 1, len(x))
    return FitResult('exponential_conditional', True, 1, ll, aic, bic, {'lambda': lam, 'xmin': xmin})


def fit_pareto(x: np.ndarray, xmin: float) -> FitResult:
    logs = np.log(x / xmin)
    denom = float(np.sum(logs))
    if denom <= 0:
        return fail('pareto_power_law', 1, 'nonpositive_log_denominator')
    alpha = 1.0 + len(x) / denom
    ll = float(np.sum(math.log(alpha - 1.0) + (alpha - 1.0) * math.log(xmin) - alpha * np.log(x)))
    aic, bic = info_criteria(ll, 1, len(x))
    return FitResult('pareto_power_law', True, 1, ll, aic, bic, {'alpha': alpha, 'xmin': xmin})


def fit_weibull_conditional(x: np.ndarray, xmin: float) -> FitResult:
    # Standard Weibull conditioned on x >= xmin.
    def nll(theta):
        log_k, log_scale = theta
        k = math.exp(log_k)
        scale = math.exp(log_scale)
        if not np.isfinite(k) or not np.isfinite(scale) or k <= 0 or scale <= 0:
            return 1e300
        z = x / scale
        zmin = xmin / scale
        # log f(x) - log S(xmin), S = exp(-zmin^k)
        logpdf = math.log(k) - k * math.log(scale) + (k - 1.0) * np.log(x) - z ** k + zmin ** k
        val = -float(np.sum(logpdf))
        return val if np.isfinite(val) else 1e300

    starts = [
        (math.log(1.0), math.log(max(np.mean(x), xmin, 1e-6))),
        (math.log(0.6), math.log(max(np.median(x), xmin, 1e-6))),
        (math.log(1.5), math.log(max(np.mean(x), xmin, 1e-6))),
    ]
    best = None
    for st in starts:
        res = minimize(nll, np.array(st), method='Nelder-Mead', options={'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not best.success or not np.isfinite(best.fun):
        return fail('weibull_conditional', 2, f'opt_failed: {getattr(best, "message", "none")}')
    k = math.exp(float(best.x[0]))
    scale = math.exp(float(best.x[1]))
    ll = -float(best.fun)
    aic, bic = info_criteria(ll, 2, len(x))
    return FitResult('weibull_conditional', True, 2, ll, aic, bic, {'shape': k, 'scale': scale, 'xmin': xmin})


def fit_tempered_power(x: np.ndarray, xmin: float) -> FitResult:
    # f(x) ∝ x^-alpha exp(-lambda x), conditioned on x >= xmin.
    cache = {}

    def log_norm(alpha: float, lam: float) -> float:
        key = (round(alpha, 10), round(lam, 14))
        if key in cache:
            return cache[key]
        def integrand(t):
            return (t ** (-alpha)) * math.exp(-lam * t)
        try:
            z, err = quad(integrand, xmin, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
        except Exception:
            z = 0.0
        if not np.isfinite(z) or z <= 0:
            out = float('inf')
        else:
            out = math.log(z)
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
        ll = -alpha * np.sum(np.log(x)) - lam * np.sum(x) - len(x) * lz
        val = -float(ll)
        return val if np.isfinite(val) else 1e300

    mean_x = max(float(np.mean(x)), xmin, 1e-6)
    starts = []
    for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for lam in [1.0 / mean_x, 0.1 / mean_x, 1.0 / max(float(np.max(x)), mean_x)]:
            starts.append((a, math.log(max(lam, 1e-12))))
    best = None
    bounds = [(0.01, 10.0), (-30.0, 5.0)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for st in starts:
            res = minimize(nll, np.array(st), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
            if best is None or res.fun < best.fun:
                best = res
    if best is None or not np.isfinite(best.fun):
        return fail('tempered_power_law', 2, 'opt_failed')
    alpha = float(best.x[0])
    lam = math.exp(float(best.x[1]))
    ll = -float(best.fun)
    aic, bic = info_criteria(ll, 2, len(x))
    return FitResult('tempered_power_law', True, 2, ll, aic, bic, {'alpha': alpha, 'lambda': lam, 'xmin': xmin})


def fit_biexponential(x: np.ndarray, xmin: float) -> FitResult:
    # Mixture of two standard exponentials conditioned on x >= xmin.
    def nll(theta):
        p_raw, log_l1, log_l2 = theta
        p = 1.0 / (1.0 + math.exp(-p_raw))
        l1 = math.exp(log_l1)
        l2 = math.exp(log_l2)
        if not (0 < p < 1) or l1 <= 0 or l2 <= 0:
            return 1e300
        a = math.log(p) + math.log(l1) - l1 * x
        b = math.log1p(-p) + math.log(l2) - l2 * x
        denom = math.log(p * math.exp(-l1 * xmin) + (1.0 - p) * math.exp(-l2 * xmin) + EPS)
        ll = float(np.sum(np.maximum(a, b) + np.log(np.exp(a - np.maximum(a, b)) + np.exp(b - np.maximum(a, b))) - denom))
        val = -ll
        return val if np.isfinite(val) else 1e300

    mean_x = max(float(np.mean(x)), xmin, 1e-6)
    starts = [
        (0.0, math.log(1.0 / mean_x), math.log(0.1 / mean_x)),
        (0.0, math.log(2.0 / mean_x), math.log(0.2 / mean_x)),
        (1.0, math.log(1.0 / mean_x), math.log(0.05 / mean_x)),
    ]
    best = None
    bounds = [(-10.0, 10.0), (-30.0, 5.0), (-30.0, 5.0)]
    for st in starts:
        res = minimize(nll, np.array(st), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        return fail('biexponential_conditional', 3, 'opt_failed')
    p = 1.0 / (1.0 + math.exp(-float(best.x[0])))
    l1 = math.exp(float(best.x[1]))
    l2 = math.exp(float(best.x[2]))
    ll = -float(best.fun)
    aic, bic = info_criteria(ll, 3, len(x))
    return FitResult('biexponential_conditional', True, 3, ll, aic, bic, {'p': p, 'lambda1': l1, 'lambda2': l2, 'xmin': xmin})


def candidate_files():
    # Pair survival: central reviewer issue for f_soft/f_hard power-law-like behavior.
    for anion in ANIONS:
        for temp in TEMPS:
            for state in ['soft', 'hard']:
                path = OLD_CODE / 'classification' / 'event_collect' / 'event#2(Pair_breaking;survival)' / state / anion / temp / 'data.txt'
                yield 'pair_survival', anion, temp, state, path
    # Total shell-change inter-event distribution: motivates bursty/heavy-tailed event statistics.
    for anion in ANIONS:
        for temp in TEMPS:
            path = OLD_CODE / 'classification' / 'event_collect' / 'event#1(Shell_change;interevent)' / 'total' / anion / temp / 'total.txt'
            yield 'shell_change_interevent_total', anion, temp, 'total', path


def fit_all_for_dataset(data: np.ndarray, xmin: float):
    x = data[data >= xmin]
    if len(x) < MIN_N:
        return [fail('all_models', 0, f'n<{MIN_N}')]
    results = [
        fit_exponential(x, xmin),
        fit_weibull_conditional(x, xmin),
        fit_pareto(x, xmin),
        fit_tempered_power(x, xmin),
        fit_biexponential(x, xmin),
    ]
    return results


def main():
    rows = []
    for dataset, anion, temp, state, path in candidate_files():
        data = positive_data(path)
        for window_name, q in WINDOWS.items():
            x, xmin = select_window(data, q)
            xmax = float(np.max(x)) if len(x) else float('nan')
            fits = fit_all_for_dataset(data, xmin)
            valid = [r for r in fits if r.ok]
            best_aic = min((r.aic for r in valid), default=float('inf'))
            best_bic = min((r.bic for r in valid), default=float('inf'))
            for r in fits:
                rows.append({
                    'dataset': dataset,
                    'anion': anion,
                    'T': temp,
                    'state': state,
                    'window': window_name,
                    'qmin': q,
                    'xmin': xmin,
                    'xmax_observed': xmax,
                    'n': len(x),
                    'model': r.model,
                    'ok': r.ok,
                    'n_params': r.n_params,
                    'loglik': r.loglik,
                    'AIC': r.aic,
                    'BIC': r.bic,
                    'delta_AIC': r.aic - best_aic if np.isfinite(best_aic) else float('nan'),
                    'delta_BIC': r.bic - best_bic if np.isfinite(best_bic) else float('nan'),
                    'params': ';'.join(f'{k}={v:.8g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in r.params.items()),
                    'path': str(path),
                    'note': r.note,
                })
    out_csv = OUT_DIR / 'model_comparison_all.csv'
    with out_csv.open('w', newline='') as f:
        fieldnames = ['dataset', 'anion', 'T', 'state', 'window', 'qmin', 'xmin', 'xmax_observed', 'n', 'model', 'ok', 'n_params', 'loglik', 'AIC', 'BIC', 'delta_AIC', 'delta_BIC', 'params', 'path', 'note']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Best-model summary.
    groups = {}
    for row in rows:
        if str(row['ok']) != 'True':
            continue
        key = (row['dataset'], row['anion'], row['T'], row['state'], row['window'])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, vals in sorted(groups.items()):
        vals_aic = sorted(vals, key=lambda r: float(r['AIC']))
        vals_bic = sorted(vals, key=lambda r: float(r['BIC']))
        top = vals_aic[0]
        second = vals_aic[1] if len(vals_aic) > 1 else None
        summary_rows.append({
            'dataset': key[0], 'anion': key[1], 'T': key[2], 'state': key[3], 'window': key[4],
            'n': top['n'], 'xmin': top['xmin'], 'xmax_observed': top['xmax_observed'],
            'best_AIC_model': top['model'], 'best_AIC': top['AIC'],
            'second_AIC_model': second['model'] if second else '',
            'delta_AIC_second': float(second['AIC']) - float(top['AIC']) if second else '',
            'best_BIC_model': vals_bic[0]['model'], 'best_BIC': vals_bic[0]['BIC'],
        })
    out_summary_csv = OUT_DIR / 'model_comparison_best_models.csv'
    with out_summary_csv.open('w', newline='') as f:
        fieldnames = ['dataset', 'anion', 'T', 'state', 'window', 'n', 'xmin', 'xmax_observed', 'best_AIC_model', 'best_AIC', 'second_AIC_model', 'delta_AIC_second', 'best_BIC_model', 'best_BIC']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Markdown interpretation helper.
    out_md = SUMMARY_DIR / 'model_comparison_summary.md'
    with out_md.open('w') as f:
        f.write('# Model-comparison summary\n\n')
        f.write('Generated by `compare_models.py`. Existing old-project duration files were read; outputs were written only under the revision folder.\n\n')
        f.write('Models compared: conditional exponential, conditional Weibull/stretched-exponential, Pareto power law, tempered power law, and conditional biexponential mixture.\n\n')
        f.write('Two windows are reported: `full` uses all positive durations; `tail_q75` fits only the upper quartile of each dataset. The tail window is included as a diagnostic fallback because the manuscript claims finite-window power-law-like tail behavior, not a pure full-distribution power law.\n\n')
        for dataset in ['pair_survival', 'shell_change_interevent_total']:
            f.write(f'## {dataset}\n\n')
            for window in WINDOWS:
                subset = [r for r in summary_rows if r['dataset'] == dataset and r['window'] == window]
                counts = {}
                for r in subset:
                    counts[r['best_AIC_model']] = counts.get(r['best_AIC_model'], 0) + 1
                f.write(f'### Window: {window}\n\n')
                f.write('Best-AIC model counts:\n\n')
                for model, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    f.write(f'- {model}: {count}\n')
                f.write('\n')
                f.write('| anion | T | state | n | xmin | best AIC | second | delta AIC | best BIC |\n')
                f.write('|---|---:|---|---:|---:|---|---|---:|---|\n')
                for r in subset:
                    f.write(f"| {r['anion']} | {r['T']} | {r['state']} | {r['n']} | {float(r['xmin']):.4g} | {r['best_AIC_model']} | {r['second_AIC_model']} | {float(r['delta_AIC_second']):.3g} | {r['best_BIC_model']} |\n")
                f.write('\n')
        f.write('## How to use if results are messy\n\n')
        f.write('- If stretched-exponential or biexponential models win for the full distribution, soften the manuscript claim: the full survival distribution is not uniquely power-law.\n')
        f.write('- If power-law or tempered-power-law models are competitive mainly in `tail_q75`, describe the result as finite-window tail behavior rather than an asymptotic law.\n')
        f.write('- If no heavy-tail model wins, use the analysis as a transparent negative control and revise the language to non-Poissonian / heterogeneous relaxation rather than power-law behavior.\n')
        f.write('- In all cases, avoid claiming universal exponents.\n')

    print(out_csv)
    print(out_summary_csv)
    print(out_md)

if __name__ == '__main__':
    main()
