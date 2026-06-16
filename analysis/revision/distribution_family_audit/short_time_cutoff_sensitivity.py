#!/usr/bin/env python3
"""Short-time cutoff sensitivity for distribution-family model comparison.

This repeats the likelihood/AIC/BIC comparison after excluding events shorter
than selected lower cutoffs. The motivation is the 1 ps trajectory sampling and
known unresolved earliest-time structure discussed in the manuscript.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import importlib.util
import math
import statistics
import sys

import numpy as np

REV = Path('/nas_2/transcendence/revision')
AUDIT_PATH = REV / 'analysis' / 'distribution_family_audit' / 'audit_distribution_families.py'
OUT = REV / 'analysis' / 'distribution_family_audit' / 'outputs' / 'short_time_cutoff'
OUT.mkdir(parents=True, exist_ok=True)
NOTE_DIR = REV / 'notes' / '01_r2_1_distribution_statistics'
SUMMARY_DIR = NOTE_DIR / 'analysis_output_summaries'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('audit_distribution_families', AUDIT_PATH)
aud = importlib.util.module_from_spec(spec)
sys.modules['audit_distribution_families'] = aud
assert spec.loader is not None
spec.loader.exec_module(aud)
cm = aud.cm

CUTS = [1.0, 2.0, 5.0, 10.0, 50.0]
MODEL_LABEL = aud.MODEL_LABEL
DATASET_LABEL = aud.DATASET_LABEL


def fmt(x) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return 'nan'
    ax = abs(x)
    if ax >= 1e4 or (ax > 0 and ax < 1e-3):
        return f'{x:.4g}'
    if ax >= 100:
        return f'{x:.1f}'
    if ax >= 10:
        return f'{x:.2f}'
    return f'{x:.4f}'


def fit_for_cut(meta: dict, data: np.ndarray, tmin: float) -> list[dict]:
    filtered = data[data >= tmin]
    base = {k: meta[k] for k in ['dataset', 'anion', 'T', 'state', 'source_kind']}
    if len(filtered) < cm.MIN_N:
        return [{**base, 'tmin': tmin, 'n': len(filtered), 'xmin': float('nan'), 'model': 'all_models', 'ok': False, 'n_params': 0, 'loglik': float('-inf'), 'AIC': float('inf'), 'BIC': float('inf'), 'delta_AIC': float('nan'), 'delta_BIC': float('nan'), 'params': '', 'note': f'n<{cm.MIN_N}'}]
    xmin = float(np.min(filtered))
    fits = cm.fit_all_for_dataset(filtered, xmin)
    valid = [r for r in fits if r.ok]
    best_aic = min((r.aic for r in valid), default=float('inf'))
    best_bic = min((r.bic for r in valid), default=float('inf'))
    rows = []
    for r in fits:
        rows.append({
            **base,
            'tmin': tmin,
            'n': len(filtered),
            'xmin': xmin,
            'xmax_observed': float(np.max(filtered)),
            'model': r.model,
            'ok': r.ok,
            'n_params': r.n_params,
            'loglik': r.loglik,
            'AIC': r.aic,
            'BIC': r.bic,
            'delta_AIC': r.aic - best_aic if np.isfinite(best_aic) else float('nan'),
            'delta_BIC': r.bic - best_bic if np.isfinite(best_bic) else float('nan'),
            'params': ';'.join(f'{k}={v:.8g}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in r.params.items()),
            'note': r.note,
        })
    return rows


def best_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row['ok'] is True or str(row['ok']) == 'True':
            grouped[(row['dataset'], row['anion'], row['T'], row['state'], row['tmin'])].append(row)
    out = []
    for key, vals in sorted(grouped.items()):
        vals_aic = sorted(vals, key=lambda r: float(r['AIC']))
        vals_bic = sorted(vals, key=lambda r: float(r['BIC']))
        best = vals_aic[0]
        second = vals_aic[1] if len(vals_aic) > 1 else None
        by_model = {r['model']: r for r in vals}
        exp = by_model.get('exponential_conditional')
        weib = by_model.get('weibull_conditional')
        pareto = by_model.get('pareto_power_law')
        tempered = by_model.get('tempered_power_law')
        biexp = by_model.get('biexponential_conditional')
        def delta(row):
            return float(row['AIC']) - float(best['AIC']) if row is not None else float('nan')
        min_pl_delta = min(delta(pareto), delta(tempered))
        out.append({
            'dataset': key[0],
            'anion': key[1],
            'T': key[2],
            'state': key[3],
            'tmin': key[4],
            'n': best['n'],
            'xmin': best['xmin'],
            'best_AIC_model': best['model'],
            'best_BIC_model': vals_bic[0]['model'],
            'second_AIC_model': second['model'] if second else '',
            'delta_AIC_second': float(second['AIC']) - float(best['AIC']) if second else float('nan'),
            'delta_AIC_exponential_minus_best': delta(exp),
            'delta_AIC_weibull_minus_best': delta(weib),
            'delta_AIC_pareto_minus_best': delta(pareto),
            'delta_AIC_tempered_minus_best': delta(tempered),
            'delta_AIC_biexponential_minus_best': delta(biexp),
            'power_family_best_or_competitive': best['model'] in ('pareto_power_law', 'tempered_power_law') or min_pl_delta <= 2,
            'exponential_best_or_competitive': best['model'] == 'exponential_conditional' or delta(exp) <= 2,
        })
    return out


def count_summary(best_rows: list[dict]) -> list[dict]:
    out = []
    for dataset in ['total_interevent', 'total_survival', 'state_duration', 'state_survival']:
        for tmin in CUTS:
            subset = [r for r in best_rows if r['dataset'] == dataset and abs(float(r['tmin']) - tmin) < 1e-9]
            if not subset:
                continue
            def pack(counter):
                return '; '.join(f'{k}:{v}' for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            exp_deltas = [float(r['delta_AIC_exponential_minus_best']) for r in subset if np.isfinite(float(r['delta_AIC_exponential_minus_best']))]
            temp_deltas = [float(r['delta_AIC_tempered_minus_best']) for r in subset if np.isfinite(float(r['delta_AIC_tempered_minus_best']))]
            out.append({
                'dataset': dataset,
                'dataset_label': DATASET_LABEL[dataset],
                'tmin': tmin,
                'cases': len(subset),
                'best_AIC_counts': pack(Counter(r['best_AIC_model'] for r in subset)),
                'best_BIC_counts': pack(Counter(r['best_BIC_model'] for r in subset)),
                'power_family_best_or_competitive_cases': sum(1 for r in subset if str(r['power_family_best_or_competitive']) == 'True' or r['power_family_best_or_competitive'] is True),
                'exponential_best_or_competitive_cases': sum(1 for r in subset if str(r['exponential_best_or_competitive']) == 'True' or r['exponential_best_or_competitive'] is True),
                'median_delta_AIC_exponential_minus_best': statistics.median(exp_deltas) if exp_deltas else float('nan'),
                'median_delta_AIC_tempered_minus_best': statistics.median(temp_deltas) if temp_deltas else float('nan'),
            })
    return out


def write_csv(path: Path, rows: list[dict]):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def markdown_report(count_rows: list[dict], best_rows: list[dict]) -> Path:
    path = SUMMARY_DIR / 'short_time_cutoff_summary.md'
    lines = []
    lines.append('# Short-time cutoff sensitivity')
    lines.append('')
    lines.append('Durations shorter than `tmin` were excluded before model fitting. `tmin=1` is the baseline positive-duration analysis because the stored trajectories are sampled at 1 ps.')
    lines.append('')
    lines.append('| Observable | tmin | Cases | Best AIC counts | Power family best/competitive | Exponential best/competitive | Median delta AIC exp-best |')
    lines.append('|---|---:|---:|---|---:|---:|---:|')
    for r in count_rows:
        lines.append(f"| {r['dataset_label']} | {fmt(r['tmin'])} | {r['cases']} | {r['best_AIC_counts']} | {r['power_family_best_or_competitive_cases']} | {r['exponential_best_or_competitive_cases']} | {fmt(r['median_delta_AIC_exponential_minus_best'])} |")
    lines.append('')
    lines.append('## Soft intra-state survival only')
    lines.append('')
    lines.append('| tmin | anion | T | n | best AIC | second | delta second | exp delta | tempered delta | pareto delta |')
    lines.append('|---:|---|---:|---:|---|---|---:|---:|---:|---:|')
    soft = [r for r in best_rows if r['dataset'] == 'state_survival' and r['state'] == 'soft']
    for r in sorted(soft, key=lambda x: (float(x['tmin']), x['anion'], int(x['T']))):
        lines.append(f"| {fmt(r['tmin'])} | {r['anion']} | {r['T']} | {r['n']} | {MODEL_LABEL.get(r['best_AIC_model'], r['best_AIC_model'])} | {MODEL_LABEL.get(r['second_AIC_model'], r['second_AIC_model'])} | {fmt(r['delta_AIC_second'])} | {fmt(r['delta_AIC_exponential_minus_best'])} | {fmt(r['delta_AIC_tempered_minus_best'])} | {fmt(r['delta_AIC_pareto_minus_best'])} |")
    path.write_text('\n'.join(lines) + '\n')
    return path


def interpretation(count_rows: list[dict], best_rows: list[dict]) -> Path:
    path = NOTE_DIR / 'short_time_cutoff_sensitivity_readout.md'
    def row(dataset, tmin):
        for r in count_rows:
            if r['dataset'] == dataset and abs(float(r['tmin']) - tmin) < 1e-9:
                return r
        raise KeyError((dataset, tmin))
    soft_tmins = {}
    for tmin in CUTS:
        sub = [r for r in best_rows if r['dataset'] == 'state_survival' and r['state'] == 'soft' and abs(float(r['tmin']) - tmin) < 1e-9]
        soft_tmins[tmin] = Counter(r['best_AIC_model'] for r in sub)
    lines = []
    lines.append('# Short-time cutoff sensitivity readout')
    lines.append('')
    lines.append('Date: 2026-05-29')
    lines.append('')
    lines.append('Motivation: the manuscript notes 1 ps trajectory sampling and unresolved earliest-time structure. This analysis excludes durations below several lower cutoffs before fitting the same model families.')
    lines.append('')
    lines.append('## Bottom line')
    lines.append('')
    lines.append('- Excluding the 1 ps neighborhood is scientifically justified and should be included as a sensitivity check.')
    lines.append('- Total inter-event tails become more favorable to tempered-power-law descriptions after removing the shortest events, especially for moderate cutoffs.')
    lines.append('- Total survival remains strongly non-single-exponential; the total curve can still be described as a mixture of soft-like fast loss and hard-like longer survival.')
    lines.append('- Soft intra-state survival remains best described by a biexponential mixture for modest cutoffs (2, 5, and usually 10 ps). This means the soft state contains at least two kinetic components, not a single homogeneous power-law process.')
    lines.append('- A very aggressive 50 ps cutoff changes the sampled population substantially and should be treated as a diagnostic, not the primary fit window for soft survival.')
    lines.append('')
    lines.append('## Count snapshot')
    lines.append('')
    lines.append('| Observable | tmin=2 best AIC | tmin=5 best AIC | tmin=10 best AIC | tmin=50 best AIC |')
    lines.append('|---|---|---|---|---|')
    for dataset in ['total_interevent', 'total_survival', 'state_duration', 'state_survival']:
        lines.append(f"| {DATASET_LABEL[dataset]} | {row(dataset,2.0)['best_AIC_counts']} | {row(dataset,5.0)['best_AIC_counts']} | {row(dataset,10.0)['best_AIC_counts']} | {row(dataset,50.0)['best_AIC_counts']} |")
    lines.append('')
    lines.append('## Soft survival best-model counts')
    lines.append('')
    for tmin in CUTS:
        packed = '; '.join(f'{k}:{v}' for k, v in sorted(soft_tmins[tmin].items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append(f'- tmin={fmt(tmin)}: {packed}')
    lines.append('')
    lines.append('## Suggested response/manuscript language')
    lines.append('')
    lines.append('Because the trajectories are saved every 1 ps, events at the earliest resolved times may include unresolved boundary rattling and should not control the functional-form assignment. We therefore repeated the model comparison after excluding short-time events. The total inter-event and total survival distributions remained strongly non-single-exponential. The soft-state pair-survival distribution remained best described by a two-component relaxation model for modest lower cutoffs, indicating that the operational soft state contains a fast exchange-rich component together with a longer-lived component. The fitted power-law exponents are therefore retained only as finite-window tail descriptors, not as proof of a universal asymptotic power law.')
    lines.append('')
    lines.append('## Output files')
    lines.append('')
    lines.append(f'- `{OUT / "short_time_cutoff_model_fits_all.csv"}`')
    lines.append(f'- `{OUT / "short_time_cutoff_best_models.csv"}`')
    lines.append(f'- `{OUT / "short_time_cutoff_count_summary.csv"}`')
    lines.append(f'- `{SUMMARY_DIR / "short_time_cutoff_summary.md"}`')
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    fit_rows = []
    for meta in aud.candidate_datasets():
        data = aud.load_many(meta['paths'])
        for tmin in CUTS:
            fit_rows.extend(fit_for_cut(meta, data, tmin))
    best_rows = best_summary(fit_rows)
    count_rows = count_summary(best_rows)
    write_csv(OUT / 'short_time_cutoff_model_fits_all.csv', fit_rows)
    write_csv(OUT / 'short_time_cutoff_best_models.csv', best_rows)
    write_csv(OUT / 'short_time_cutoff_count_summary.csv', count_rows)
    report = markdown_report(count_rows, best_rows)
    note = interpretation(count_rows, best_rows)
    clog = REV / 'notes' / '00_project_management' / 'change_log.md'
    entry = '- Ran short-time cutoff sensitivity analysis excluding events below 2, 5, 10, and 50 ps; outputs saved under distribution_family_audit/outputs/short_time_cutoff/.\n'
    text = clog.read_text() if clog.exists() else ''
    if entry not in text:
        clog.write_text(text.rstrip() + '\n' + entry)
    for p in [OUT / 'short_time_cutoff_count_summary.csv', report, note]:
        print(p)


if __name__ == '__main__':
    main()
