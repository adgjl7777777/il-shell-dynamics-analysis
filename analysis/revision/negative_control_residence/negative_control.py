#!/usr/bin/env python3
"""Negative control for the R2-1 model-selection pipeline (INTERNAL CHECK).

Question it answers: "Does the AIC/BIC model-selection pipeline intrinsically
favor more complex / more flexible model families, so that a single exponential
could never win even when the data really are exponential?"

Method (scoped to STATE RESIDENCE DURATIONS only, where the exponential-vs-complex
call is genuinely close):
  1. For each residence-duration dataset, fit the SAME conditional exponential the
     real pipeline uses (x = xmin + Exp(lambda), xmin = tmin).
  2. Generate synthetic samples of the SAME size n from that fitted exponential.
  3. Refit ALL FIVE models with the IDENTICAL pipeline (compare_models.py) and pick
     the best model by BIC (BIC is emphasized: its ln(n) complexity penalty is the
     stricter one).
  4. Repeat R times; report the fraction of replicates in which BIC recovers the
     exponential.

If exponential-generated data is recovered as exponential at a high rate, the
pipeline is NOT biased toward complex families, so the rejection of the single
exponential on the real survival data is meaningful rather than an artifact of
parameter count.

This script reads old-project residence-duration files and writes ONLY under the
revision analysis folder. It does not modify the manuscript.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Reuse the EXACT real pipeline (model definitions, BIC, MIN_N).
sys.path.insert(0, str(Path('/nas_2/transcendence/revision/analysis/powerlaw_model_comparison')))
import compare_models as cm  # noqa: E402

OLD_CODE = Path('/nas_2/transcendence/il_paper/code')
OUT = Path('/nas_2/transcendence/revision/analysis/negative_control_residence/outputs')
OUT.mkdir(parents=True, exist_ok=True)

ANIONS = ['fsi', 'tfsi', 'beti']
TEMPS = ['298', '353', '373', '423']
STATES = ['soft', 'hard']
TMINS = [1.0, 2.0]      # 2.0 matches the manuscript residence-duration claim; 1.0 is the baseline
R = 200                 # synthetic replicates per condition
SEED = 20260612


def residence_path(state, anion, temp):
    return OLD_CODE / 'classification' / 'event_collect' / 'soft_hard_duration' / state / anion / temp / 'data.txt'


def best_bic_model(x, xmin):
    fits = cm.fit_all_for_dataset(x, xmin)
    valid = [f for f in fits if f.ok]
    if not valid:
        return None
    return min(valid, key=lambda f: f.bic).model


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for tmin in TMINS:
        for state in STATES:
            for anion in ANIONS:
                for temp in TEMPS:
                    p = residence_path(state, anion, temp)
                    if not p.exists():
                        continue
                    data = cm.positive_data(p)
                    x_real = data[data >= tmin]
                    n = len(x_real)
                    if n < cm.MIN_N:
                        rows.append(dict(tmin=tmin, state=state, anion=anion, T=temp, n=n,
                                         lam=float('nan'), recovery_bic=float('nan'),
                                         note=f'n<{cm.MIN_N}_skipped'))
                        continue
                    # Fit the conditional exponential the pipeline uses: x = tmin + Exp(lam)
                    fexp = cm.fit_exponential(x_real, tmin)
                    lam = fexp.params['lambda']
                    # Negative control: generate from that exponential, refit, count BIC winners.
                    exp_wins = 0
                    counter = {}
                    for _ in range(R):
                        x_syn = tmin + rng.exponential(1.0 / lam, size=n)
                        m = best_bic_model(x_syn, tmin)
                        counter[m] = counter.get(m, 0) + 1
                        if m == 'exponential_conditional':
                            exp_wins += 1
                    rows.append(dict(
                        tmin=tmin, state=state, anion=anion, T=temp, n=n,
                        lam=lam, recovery_bic=exp_wins / R,
                        winners=';'.join(f'{k}:{v}' for k, v in sorted(counter.items(), key=lambda kv: -kv[1])),
                        note='',
                    ))
                    print(f'tmin={tmin} {state:4s} {anion:4s} {temp}  n={n:5d}  BIC-exp recovery={exp_wins/R:.3f}', flush=True)

    # Per-condition CSV
    out_csv = OUT / 'negative_control_recovery.csv'
    with out_csv.open('w', newline='') as f:
        cols = ['tmin', 'state', 'anion', 'T', 'n', 'lam', 'recovery_bic', 'winners', 'note']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate summary
    summ = OUT / 'negative_control_summary.md'
    with summ.open('w') as f:
        f.write('# Negative control — residence-duration model selection (internal)\n\n')
        f.write(f'R = {R} synthetic replicates per condition, seed = {SEED}. Synthetic data drawn '
                'from the fitted conditional exponential; model picked by **BIC** with the identical '
                'five-model pipeline (`compare_models.py`).\n\n')
        for tmin in TMINS:
            valid = [r for r in rows if r['tmin'] == tmin and np.isfinite(r.get('recovery_bic', float('nan')))]
            if not valid:
                continue
            recs = [r['recovery_bic'] for r in valid]
            f.write(f'## tmin = {tmin:g} ps\n\n')
            f.write(f'- conditions used: {len(valid)} (n >= {cm.MIN_N})\n')
            f.write(f'- mean BIC exponential-recovery: **{np.mean(recs):.3f}**\n')
            f.write(f'- min / median / max: {np.min(recs):.3f} / {np.median(recs):.3f} / {np.max(recs):.3f}\n')
            f.write(f'- conditions with recovery >= 0.90: {sum(rc >= 0.90 for rc in recs)}/{len(recs)}\n\n')
            f.write('| state | anion | T | n | BIC exp-recovery | top synthetic winners |\n')
            f.write('|---|---|---:|---:|---:|---|\n')
            for r in sorted(valid, key=lambda r: (r['state'], r['anion'], int(r['T']))):
                f.write(f"| {r['state']} | {r['anion']} | {r['T']} | {r['n']} | {r['recovery_bic']:.3f} | {r.get('winners','')} |\n")
            f.write('\n')
    print('\nWROTE:', out_csv)
    print('WROTE:', summ)


if __name__ == '__main__':
    main()
