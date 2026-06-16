#!/usr/bin/env python3
"""eta-sensitivity of (a) the survival power-law exponent beta and (b) the AIC/BIC
best-fitting model family, for the state-resolved pair survival (INTERNAL / review-only).

For each threshold multiplier eta the soft/hard classification already exists
(result/<anion>/<state>/<T>/<eta>_<ion>.txt). We replicate the soft_hard_split logic
to assign each pair event to intra-soft (f_soft) or intra-hard (f_hard) survival, then:
  - fit a Pareto power-law exponent beta to f_soft/f_hard (cheap -> all eta), and
  - run the 5-model AIC/BIC comparison (compare_models.py) on a reduced eta grid.

Goal: show whether beta and the model selection are stable across eta, i.e. whether the
two-state survival description is robust to the operational threshold (eta=1 is the
mean-inter-event-time canonical choice, not a fitted optimum).
"""
from __future__ import annotations
import sys, csv, bisect, math
from pathlib import Path
import numpy as np

sys.path.insert(0, '/nas_2/transcendence/revision/analysis/powerlaw_model_comparison')
import compare_models as cm

CLS = Path('/nas_2/transcendence/il_paper/code/classification')
OUT = Path('/nas_2/transcendence/revision/analysis/eta_beta_model_sensitivity/outputs')
OUT.mkdir(parents=True, exist_ok=True)
ANIONS = ['fsi', 'tfsi', 'beti']
TEMPS = ['298', '353', '373', '423']
ALL_ETA = ['0.1','0.3','0.5','0.7','0.8','0.9','1.0','1.1','1.2','1.3','1.5','2.0','3.0','5.0','10.0']
BIC_ETA = ['0.5','0.7','0.9','1.0','1.2','1.5','2.0','3.0']   # reduced grid for the 5-model BIC
MIN_N = 30


def load_intervals(anion, T, eta, ion):
    iv = []
    for state, val in [('soft', 0), ('hard', 1)]:
        f = CLS / 'result' / anion / state / T / f'{eta}_{ion}.txt'
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            nums = [int(x) for x in line.split() if x.lstrip('-').isdigit()]
            if nums:
                iv.append((nums[0], nums[-1], val))   # (start, end, state)
    iv.sort()
    return iv


def state_at(iv, starts, frame):
    j = bisect.bisect_right(starts, frame) - 1
    if 0 <= j < len(iv) and iv[j][0] <= frame <= iv[j][1]:
        return iv[j][2]
    return None


def survival_by_state(anion, T, eta):
    """Return (f_soft_durations, f_hard_durations) pooled over 5 ions."""
    f_soft, f_hard = [], []
    for ion in range(5):
        iv = load_intervals(anion, T, eta, ion)
        if iv is None:
            continue
        starts = [a for a, _, _ in iv]
        pf = CLS / 'pair' / anion / T / f'{ion}.txt'
        if not pf.exists():
            continue
        ev = np.loadtxt(pf)
        if ev.ndim == 1:
            ev = ev.reshape(1, -1)
        for s, e in ev:
            si = state_at(iv, starts, int(s)); sf = state_at(iv, starts, int(e))
            if si is None or sf is None:
                continue
            dur = e - s
            if dur <= 0:
                continue
            if si == 0 and sf == 0:
                f_soft.append(dur)
            elif si == 1 and sf == 1:
                f_hard.append(dur)
    return np.array(f_soft, float), np.array(f_hard, float)


def pareto_beta(x):
    if len(x) < MIN_N:
        return float('nan'), len(x)
    xmin = float(np.min(x))
    r = cm.fit_pareto(x[x >= xmin], xmin)
    # survival exponent beta ~ alpha-1 for a Pareto density alpha
    return (r.params['alpha'] - 1.0) if r.ok else float('nan'), len(x)


def main():
    beta_rows, bic_rows = [], []
    for anion in ANIONS:
        for T in TEMPS:
            for eta in ALL_ETA:
                fs, fh = survival_by_state(anion, T, eta)
                bs, ns = pareto_beta(fs)
                bh, nh = pareto_beta(fh)
                beta_rows.append(dict(anion=anion, T=T, eta=float(eta),
                                      beta_soft=round(bs, 3), n_soft=ns,
                                      beta_hard=round(bh, 3), n_hard=nh))
                if eta in BIC_ETA:
                    for state, x in [('soft', fs), ('hard', fh)]:
                        if len(x) < MIN_N:
                            best = 'n<%d' % MIN_N
                        else:
                            xmin = float(np.min(x))
                            fits = [f for f in cm.fit_all_for_dataset(x, xmin) if f.ok]
                            best = min(fits, key=lambda f: f.bic).model if fits else 'none'
                        bic_rows.append(dict(anion=anion, T=T, eta=float(eta), state=state,
                                             n=len(x), best_BIC_model=best))
                print(f'  {anion:4s} {T} eta={eta:>4}: beta_soft={bs:.2f}(n{ns}) beta_hard={bh:.2f}(n{nh})', flush=True)

    with (OUT/'beta_vs_eta.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(beta_rows[0].keys())); w.writeheader(); w.writerows(beta_rows)
    with (OUT/'bic_model_vs_eta.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(bic_rows[0].keys())); w.writeheader(); w.writerows(bic_rows)

    # ---- summaries ----
    import statistics
    def near1(rows, key):
        # variation of beta within eta in [0.7,1.5] vs full range, per condition
        out = []
        for anion in ANIONS:
            for T in TEMPS:
                sub = [r for r in beta_rows if r['anion'] == anion and r['T'] == T and not math.isnan(r[key])]
                near = [r[key] for r in sub if 0.7 <= r['eta'] <= 1.5]
                full = [r[key] for r in sub]
                if near and full:
                    out.append((max(near)-min(near), max(full)-min(full)))
        return out

    with (OUT/'eta_sensitivity_summary.md').open('w') as f:
        f.write('# eta-sensitivity of beta and BIC model selection (internal)\n\n')
        for key, lab in [('beta_soft','soft'), ('beta_hard','hard')]:
            sp = near1(None, key)
            if sp:
                nearspread = statistics.median([a for a, _ in sp])
                fullspread = statistics.median([b for _, b in sp])
                f.write(f'- {lab}-state survival beta: median spread within eta in [0.7,1.5] = {nearspread:.2f}; '
                        f'across full eta [0.1,10] = {fullspread:.2f}.\n')
        # BIC model stability
        f.write('\n## Best-BIC model vs eta (does the selected family change with eta?)\n\n')
        from collections import Counter
        for state in ['soft', 'hard']:
            # for each condition, count distinct best models over BIC_ETA
            changes = []
            for anion in ANIONS:
                for T in TEMPS:
                    models = [r['best_BIC_model'] for r in bic_rows
                              if r['anion']==anion and r['T']==T and r['state']==state and not r['best_BIC_model'].startswith('n<')]
                    if models:
                        changes.append(len(set(models)))
            if changes:
                f.write(f'- {state}-state: across the reduced eta grid, the best-BIC family takes '
                        f'{statistics.median(changes):.0f} distinct value(s) per condition (median; '
                        f'1 = fully stable). Conditions fully stable: {sum(c==1 for c in changes)}/{len(changes)}.\n')
        f.write('\nInterpretation: small beta spread and a stable best-BIC family across eta indicate that the '
                'state-resolved survival description is robust to the operational threshold; eta=1 is the '
                'mean-inter-event-time canonical choice rather than a fitted optimum.\n')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
