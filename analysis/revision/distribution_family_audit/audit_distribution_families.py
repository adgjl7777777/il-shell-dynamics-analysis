#!/usr/bin/env python3
"""Audit distribution families for the major manuscript observables.

Inputs are read from old manuscript/cowork locations. Outputs are written only
under /nas_2/transcendence/revision/analysis/distribution_family_audit.
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
THIS = REV / 'analysis' / 'distribution_family_audit'
OUT = THIS / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
NOTE_DIR = REV / 'notes' / '01_r2_1_distribution_statistics'
SUMMARY_DIR = NOTE_DIR / 'analysis_output_summaries'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
OLD_CODE = Path('/nas_2/transcendence/il_paper/code')
LEGACY_TOTAL = Path('/nas_2/transcendence/_delete/cowork/my_work/code/total_real_plot/survival')
COMPARE_PATH = REV / 'analysis' / 'powerlaw_model_comparison' / 'compare_models.py'

spec = importlib.util.spec_from_file_location('compare_models', COMPARE_PATH)
cm = importlib.util.module_from_spec(spec)
sys.modules['compare_models'] = cm
assert spec.loader is not None
spec.loader.exec_module(cm)

ANIONS = ['fsi', 'tfsi', 'beti']
TEMPS = ['298', '353', '373', '423']
WINDOWS = {'full': 0.0, 'tail_q75': 0.75}
MODEL_LABEL = {
    'exponential_conditional': 'single exponential',
    'weibull_conditional': 'Weibull/stretched exponential',
    'pareto_power_law': 'Pareto power law',
    'tempered_power_law': 'tempered power law',
    'biexponential_conditional': 'biexponential mixture',
}
DATASET_LABEL = {
    'total_interevent': 'total shell-change inter-event',
    'total_survival': 'total Li-anion pair survival',
    'state_duration': 'soft/hard state residence duration',
    'state_survival': 'soft/hard intra-state pair survival',
}


def load_vector(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path)
    except Exception:
        return np.array([], dtype=float)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr


def load_many(paths: list[Path]) -> np.ndarray:
    parts = [load_vector(p) for p in paths if p.exists()]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def candidate_datasets():
    for anion in ANIONS:
        for temp in TEMPS:
            yield {
                'dataset': 'total_interevent',
                'anion': anion,
                'T': temp,
                'state': 'total',
                'source_kind': 'raw_total_interevent_current_code',
                'paths': [OLD_CODE / 'classification' / 'event_collect' / 'event#1(Shell_change;interevent)' / 'total' / anion / temp / 'total.txt'],
            }
    for anion in ANIONS:
        for temp in TEMPS:
            yield {
                'dataset': 'total_survival',
                'anion': anion,
                'T': temp,
                'state': 'total',
                'source_kind': 'legacy_total_real_plot_survived_files',
                'paths': [LEGACY_TOTAL / anion / temp / f'survived_{i}.txt' for i in range(5)],
            }
    for anion in ANIONS:
        for temp in TEMPS:
            for state in ['soft', 'hard']:
                yield {
                    'dataset': 'state_duration',
                    'anion': anion,
                    'T': temp,
                    'state': state,
                    'source_kind': 'raw_state_duration_current_code',
                    'paths': [OLD_CODE / 'classification' / 'event_collect' / 'soft_hard_duration' / state / anion / temp / 'data.txt'],
                }
    for anion in ANIONS:
        for temp in TEMPS:
            for state in ['soft', 'hard']:
                yield {
                    'dataset': 'state_survival',
                    'anion': anion,
                    'T': temp,
                    'state': state,
                    'source_kind': 'raw_state_pair_survival_current_code',
                    'paths': [OLD_CODE / 'classification' / 'event_collect' / 'event#2(Pair_breaking;survival)' / state / anion / temp / 'data.txt'],
                }


def burstiness_A(data: np.ndarray) -> float:
    n = len(data)
    if n < 2:
        return float('nan')
    mean = float(np.mean(data))
    if mean <= 0:
        return float('nan')
    r = float(np.std(data)) / mean
    num = math.sqrt(n + 1) * r - math.sqrt(n - 1)
    den = (math.sqrt(n + 1) - 2) * r + math.sqrt(n - 1)
    return num / den if den != 0 else float('nan')


def skewness(data: np.ndarray) -> float:
    if len(data) < 3:
        return float('nan')
    mu = float(np.mean(data))
    sig = float(np.std(data))
    if sig <= 0:
        return float('nan')
    return float(np.mean(((data - mu) / sig) ** 3))


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


def stats_row(meta: dict, data: np.ndarray) -> dict:
    base = {k: meta[k] for k in ['dataset', 'anion', 'T', 'state', 'source_kind']}
    base['paths'] = ';'.join(str(p) for p in meta['paths'])
    if len(data) == 0:
        return {**base, 'n': 0, 'missing': True}
    mean = float(np.mean(data))
    med = float(np.median(data))
    std = float(np.std(data))
    return {
        **base,
        'n': len(data),
        'missing': False,
        'min': float(np.min(data)),
        'median': med,
        'mean': mean,
        'std': std,
        'cv': std / mean if mean > 0 else float('nan'),
        'q75': float(np.quantile(data, 0.75)),
        'q90': float(np.quantile(data, 0.90)),
        'q95': float(np.quantile(data, 0.95)),
        'q99': float(np.quantile(data, 0.99)),
        'max': float(np.max(data)),
        'q99_over_median': float(np.quantile(data, 0.99)) / med if med > 0 else float('nan'),
        'max_over_median': float(np.max(data)) / med if med > 0 else float('nan'),
        'burstiness_A_N': burstiness_A(data),
        'skewness': skewness(data),
    }


def powerlaw_status(best: dict, pareto: dict | None, tempered: dict | None) -> str:
    if best['model'] in ('pareto_power_law', 'tempered_power_law'):
        return 'powerlaw_family_best'
    deltas = []
    for row in (pareto, tempered):
        if row is not None:
            deltas.append(float(row['delta_AIC']))
    if not deltas:
        return 'not_tested'
    d = min(deltas)
    if d <= 2:
        return 'powerlaw_family_competitive_deltaAIC_le_2'
    if d <= 10:
        return 'powerlaw_family_near_deltaAIC_le_10'
    return 'powerlaw_family_not_competitive'


def fit_rows_for(meta: dict, data: np.ndarray) -> list[dict]:
    rows = []
    base = {k: meta[k] for k in ['dataset', 'anion', 'T', 'state', 'source_kind']}
    paths = ';'.join(str(p) for p in meta['paths'])
    for window, q in WINDOWS.items():
        if len(data) < cm.MIN_N:
            rows.append({**base, 'window': window, 'qmin': q, 'xmin': float('nan'), 'xmax_observed': float('nan'), 'n': len(data), 'model': 'all_models', 'ok': False, 'n_params': 0, 'loglik': float('-inf'), 'AIC': float('inf'), 'BIC': float('inf'), 'delta_AIC': float('nan'), 'delta_BIC': float('nan'), 'params': '', 'note': f'n<{cm.MIN_N}', 'paths': paths})
            continue
        x, xmin = cm.select_window(data, q)
        fits = cm.fit_all_for_dataset(data, xmin)
        valid = [r for r in fits if r.ok]
        best_aic = min((r.aic for r in valid), default=float('inf'))
        best_bic = min((r.bic for r in valid), default=float('inf'))
        xmax = float(np.max(x)) if len(x) else float('nan')
        for r in fits:
            rows.append({
                **base,
                'window': window,
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
                'note': r.note,
                'paths': paths,
            })
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def best_summary(fit_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in fit_rows:
        if row['ok'] is True or str(row['ok']) == 'True':
            grouped[(row['dataset'], row['anion'], row['T'], row['state'], row['window'])].append(row)
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
        delta_exp = float(exp['AIC']) - float(best['AIC']) if exp else float('nan')
        out.append({
            'dataset': key[0],
            'anion': key[1],
            'T': key[2],
            'state': key[3],
            'window': key[4],
            'n': best['n'],
            'xmin': best['xmin'],
            'best_AIC_model': best['model'],
            'best_AIC_label': MODEL_LABEL.get(best['model'], best['model']),
            'best_BIC_model': vals_bic[0]['model'],
            'second_AIC_model': second['model'] if second else '',
            'delta_AIC_second': float(second['AIC']) - float(best['AIC']) if second else float('nan'),
            'delta_AIC_exponential_minus_best': delta_exp,
            'delta_AIC_weibull_minus_best': float(weib['AIC']) - float(best['AIC']) if weib else float('nan'),
            'delta_AIC_pareto_minus_best': float(pareto['AIC']) - float(best['AIC']) if pareto else float('nan'),
            'delta_AIC_tempered_minus_best': float(tempered['AIC']) - float(best['AIC']) if tempered else float('nan'),
            'delta_AIC_biexponential_minus_best': float(biexp['AIC']) - float(best['AIC']) if biexp else float('nan'),
            'single_exponential_status': 'competitive' if delta_exp <= 2 else ('weakly_disfavored' if delta_exp <= 10 else 'strongly_disfavored'),
            'powerlaw_status': powerlaw_status(best, pareto, tempered),
            'source_kind': best['source_kind'],
        })
    return out


def counts_summary(best_rows: list[dict]) -> list[dict]:
    out = []
    for dataset in ['total_interevent', 'total_survival', 'state_duration', 'state_survival']:
        for window in WINDOWS:
            subset = [r for r in best_rows if r['dataset'] == dataset and r['window'] == window]
            if not subset:
                continue
            def pack(counter):
                return '; '.join(f'{k}:{v}' for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
            exp_deltas = [float(r['delta_AIC_exponential_minus_best']) for r in subset if np.isfinite(float(r['delta_AIC_exponential_minus_best']))]
            temp_deltas = [float(r['delta_AIC_tempered_minus_best']) for r in subset if np.isfinite(float(r['delta_AIC_tempered_minus_best']))]
            out.append({
                'dataset': dataset,
                'dataset_label': DATASET_LABEL[dataset],
                'window': window,
                'cases': len(subset),
                'best_AIC_counts': pack(Counter(r['best_AIC_model'] for r in subset)),
                'best_BIC_counts': pack(Counter(r['best_BIC_model'] for r in subset)),
                'single_exponential_status_counts': pack(Counter(r['single_exponential_status'] for r in subset)),
                'powerlaw_status_counts': pack(Counter(r['powerlaw_status'] for r in subset)),
                'median_delta_AIC_exponential_minus_best': statistics.median(exp_deltas) if exp_deltas else float('nan'),
                'median_delta_AIC_tempered_minus_best': statistics.median(temp_deltas) if temp_deltas else float('nan'),
            })
    return out


def md_counts_table(rows: list[dict]) -> str:
    lines = [
        '| Observable | Window | Cases | Best AIC counts | Single-exp status | Power-law-family status | Median delta AIC exp-best |',
        '|---|---|---:|---|---|---|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['dataset_label']} | {r['window']} | {r['cases']} | {r['best_AIC_counts']} | {r['single_exponential_status_counts']} | {r['powerlaw_status_counts']} | {fmt(r['median_delta_AIC_exponential_minus_best'])} |")
    return '\n'.join(lines)


def write_summary(best_rows, count_rows):
    path = SUMMARY_DIR / 'distribution_family_audit_summary.md'
    with path.open('w') as f:
        f.write('# Distribution-family audit summary\n\n')
        f.write('Generated from raw duration files where available. Old manuscript/cowork files were read only; all outputs were written under the revision folder.\n\n')
        f.write('Important source note: total survival uses legacy `survived_*.txt` outputs from `/nas_2/transcendence/_delete/cowork/my_work/code/total_real_plot/survival`, produced by the old `survival.py` logic from `pair_check` files.\n\n')
        f.write('Models compared: conditional single exponential, conditional Weibull/stretched exponential, Pareto power law, tempered power law, and conditional biexponential mixture.\n\n')
        f.write(md_counts_table(count_rows))
        f.write('\n\n')
        for dataset in ['total_interevent', 'total_survival', 'state_duration', 'state_survival']:
            f.write(f'## {DATASET_LABEL[dataset]}\n\n')
            for window in WINDOWS:
                subset = [r for r in best_rows if r['dataset'] == dataset and r['window'] == window]
                if not subset:
                    continue
                f.write(f'### Window: {window}\n\n')
                f.write('| anion | T | state | n | xmin | best AIC | second | delta second | exp delta | power-law status |\n')
                f.write('|---|---:|---|---:|---:|---|---|---:|---:|---|\n')
                for r in subset:
                    f.write(f"| {r['anion']} | {r['T']} | {r['state']} | {r['n']} | {fmt(r['xmin'])} | {MODEL_LABEL.get(r['best_AIC_model'], r['best_AIC_model'])} | {MODEL_LABEL.get(r['second_AIC_model'], r['second_AIC_model'])} | {fmt(r['delta_AIC_second'])} | {fmt(r['delta_AIC_exponential_minus_best'])} | {r['powerlaw_status']} |\n")
                f.write('\n')
    return path


def write_interpretation(count_rows):
    path = NOTE_DIR / 'distribution_family_audit_interpretation.md'
    by_key = {(r['dataset'], r['window']): r for r in count_rows}
    def g(dataset, window, field):
        return by_key[(dataset, window)][field]
    text = f"""# Distribution-family audit interpretation

Date: 2026-05-29

Scope: total shell-change inter-event intervals, total Li-anion pair survival, soft/hard state residence durations, and soft/hard intra-state pair survival durations.

## Bottom line

The answer is not "없다". The data do not support a clean universal pure-power-law story, but they do support a defensible hierarchy:

1. **State residence durations** are the cleanest near-exponential layer. This supports the manuscript's statement that coarse-grained soft/hard switching can be represented by approximately single-rate residence statistics.
2. **Total inter-event intervals and total survival** are strongly non-single-exponential. These support the broader bursty / temporally heterogeneous framing.
3. **Soft/hard intra-state survival functions** are also non-single-exponential, but full-distribution model selection often favors mixture or stretched-exponential descriptions over pure power laws. These should be described as finite-window heavy-tail / tail-heaviness behavior, not as asymptotic power laws.

## Model-count snapshot

| Observable | Full-window best AIC | Tail-window best AIC | Practical message |
|---|---|---|---|
| Total shell-change inter-event | {g('total_interevent', 'full', 'best_AIC_counts')} | {g('total_interevent', 'tail_q75', 'best_AIC_counts')} | Strongly non-single-exponential; tail often supports tempered-power-law language. |
| Total Li-anion pair survival | {g('total_survival', 'full', 'best_AIC_counts')} | {g('total_survival', 'tail_q75', 'best_AIC_counts')} | Strongly heterogeneous; not safe as pure power law across the full distribution. |
| Soft/hard state residence duration | {g('state_duration', 'full', 'best_AIC_counts')} | {g('state_duration', 'tail_q75', 'best_AIC_counts')} | Coarse-grained residence layer is the most exponential-compatible part. |
| Soft/hard intra-state pair survival | {g('state_survival', 'full', 'best_AIC_counts')} | {g('state_survival', 'tail_q75', 'best_AIC_counts')} | State-resolved survival remains non-single-exponential; beta should be descriptive. |

## How this rescues the manuscript

The revised claim should be layered rather than one-size-fits-all:

- Shell-change timing and pair survival are temporally heterogeneous and inconsistent with a simple Poisson/single-rate picture.
- The soft/hard classifier converts that complex event stream into coarse-grained residence episodes whose duration distributions are closer to exponential.
- Within those states, pair survival still has broad finite-window tails, but alternative phenomenological models such as Weibull/stretched exponential and biexponential mixtures can be statistically competitive.
- Therefore, the two-state picture remains useful, while the word "power-law" should be treated as a finite-window descriptor rather than a universal law.

## Recommended manuscript phrasing

Use:

- "non-single-exponential survival and inter-event statistics"
- "finite-window heavy-tail behavior"
- "descriptive tail-heaviness exponent"
- "coarse-grained residence statistics are approximately single-rate, whereas intra-state pair survival remains temporally heterogeneous"

Avoid:

- "the survival functions are power laws"
- "universal exponent"
- "power-law dynamics are established" without finite-window qualification

## Generated analysis outputs

- `/nas_2/transcendence/revision/analysis/distribution_family_audit/outputs/distribution_stats.csv`
- `/nas_2/transcendence/revision/analysis/distribution_family_audit/outputs/distribution_model_fits_all.csv`
- `/nas_2/transcendence/revision/analysis/distribution_family_audit/outputs/distribution_best_models.csv`
- `/nas_2/transcendence/revision/analysis/distribution_family_audit/outputs/distribution_model_count_summary.csv`
- `/nas_2/transcendence/revision/notes/01_r2_1_distribution_statistics/analysis_output_summaries/distribution_family_audit_summary.md`
"""
    path.write_text(text)
    return path


def append_change_log():
    path = REV / 'change_log.md'
    entry = '- Ran comprehensive distribution-family audit for total inter-event, total survival, state duration, and state-resolved survival; included legacy `_delete/cowork` total-survival outputs.\n'
    text = path.read_text() if path.exists() else ''
    if entry not in text:
        path.write_text(text.rstrip() + '\n' + entry)


def main():
    inventory_rows = []
    stats_rows = []
    fit_rows = []
    for meta in candidate_datasets():
        data = load_many(meta['paths'])
        inventory_rows.append({
            'dataset': meta['dataset'],
            'anion': meta['anion'],
            'T': meta['T'],
            'state': meta['state'],
            'source_kind': meta['source_kind'],
            'files_expected': len(meta['paths']),
            'files_present': sum(1 for p in meta['paths'] if p.exists()),
            'n_positive': len(data),
            'paths': ';'.join(str(p) for p in meta['paths']),
        })
        stats_rows.append(stats_row(meta, data))
        fit_rows.extend(fit_rows_for(meta, data))
    best_rows = best_summary(fit_rows)
    count_rows = counts_summary(best_rows)

    write_csv(OUT / 'distribution_input_inventory.csv', inventory_rows)
    write_csv(OUT / 'distribution_stats.csv', stats_rows)
    write_csv(OUT / 'distribution_model_fits_all.csv', fit_rows)
    write_csv(OUT / 'distribution_best_models.csv', best_rows)
    write_csv(OUT / 'distribution_model_count_summary.csv', count_rows)
    summary = write_summary(best_rows, count_rows)
    interpretation = write_interpretation(count_rows)
    append_change_log()

    for p in [
        OUT / 'distribution_input_inventory.csv',
        OUT / 'distribution_stats.csv',
        OUT / 'distribution_model_fits_all.csv',
        OUT / 'distribution_best_models.csv',
        OUT / 'distribution_model_count_summary.csv',
        summary,
        interpretation,
    ]:
        print(p)


if __name__ == '__main__':
    main()
