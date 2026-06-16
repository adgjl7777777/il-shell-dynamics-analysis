#!/usr/bin/env python3
"""Additional model-validation tests for the review-only note (INTERNAL).

Test A — Chapman-Kolmogorov (CK) / Markovianity of the soft/hard 2-state process:
  Reconstruct the per-frame soft(0)/hard(1) label series for each Li ion from the
  classification episode files, build the lag-tau transition matrix T(tau), and test
  whether T(k*tau) (measured) matches T(tau)**k (Markov prediction). A small deviation
  means the coarse-grained 2-state switching is approximately Markovian at that lag.

Test B — Parameter identifiability / initial-guess stability:
  Refit the conditional biexponential to representative pair-survival datasets from many
  random initial guesses and report the spread of the maximized log-likelihood and the
  fitted parameters. Tight spread => the model is identifiable, not overfitting noise.

Reads old-project files read-only; writes only under the revision analysis folder.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('/nas_2/transcendence/revision/analysis/powerlaw_model_comparison')))
import compare_models as cm  # noqa

CLS = Path('/nas_2/transcendence/il_paper/code/classification')
OUT = Path('/nas_2/transcendence/revision/analysis/model_validation_extra/outputs')
OUT.mkdir(parents=True, exist_ok=True)
ANIONS = ['fsi', 'tfsi', 'beti']
TEMPS = ['298', '353', '373', '423']
TRAIL = '1.0'


# ---------- Test A: CK / Markovianity ----------
def state_series(anion, T, i):
    """Reconstruct per-frame soft(0)/hard(1) label array for one Li ion."""
    intervals = []
    for state, val in [('soft', 0), ('hard', 1)]:
        f = CLS / 'result' / anion / state / T / f'{TRAIL}_{i}.txt'
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            nums = [int(x) for x in line.split() if x.lstrip('-').isdigit()]
            if nums:
                intervals.append((nums[0], nums[-1], val))
    if not intervals:
        return None
    n = max(b for _, b, _ in intervals) + 1
    s = np.full(n, -1, dtype=int)
    for a, b, v in sorted(intervals):
        s[a:b + 1] = v
    return s  # may contain -1 gaps; handled by masking transitions


def transition_matrix(series_list, tau):
    """2x2 row-stochastic T(tau) pooled over ions (no cross-ion transitions)."""
    C = np.zeros((2, 2))
    for s in series_list:
        a = s[:-tau]; b = s[tau:]
        m = (a >= 0) & (b >= 0)
        a, b = a[m], b[m]
        for i in (0, 1):
            for j in (0, 1):
                C[i, j] += np.sum((a == i) & (b == j))
    row = C.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    return C / row, C.sum()


def ck_test():
    rows = []
    base_taus = [2, 5, 10]   # ps; CK compares T(k*tau) vs T(tau)^k
    ks = [2, 3]
    for anion in ANIONS:
        for T in TEMPS:
            series = [state_series(anion, T, i) for i in range(5)]
            series = [s for s in series if s is not None]
            if not series:
                continue
            for tau in base_taus:
                Ttau, _ = transition_matrix(series, tau)
                for k in ks:
                    Tk_meas, n_used = transition_matrix(series, tau * k)
                    Tk_pred = np.linalg.matrix_power(Ttau, k)
                    dev = float(np.max(np.abs(Tk_meas - Tk_pred)))  # max element deviation
                    rows.append(dict(anion=anion, T=T, tau=tau, k=k,
                                     pred_soft_to_hard=round(float(Tk_pred[0, 1]), 4),
                                     meas_soft_to_hard=round(float(Tk_meas[0, 1]), 4),
                                     pred_hard_to_soft=round(float(Tk_pred[1, 0]), 4),
                                     meas_hard_to_soft=round(float(Tk_meas[1, 0]), 4),
                                     max_abs_dev=round(dev, 4)))
    with (OUT / 'ck_test.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ---------- Test B: parameter identifiability ----------
def stability_test():
    rng = np.random.default_rng(7)
    targets = [('fsi', '298', 'soft'), ('beti', '298', 'hard'), ('tfsi', '373', 'soft')]
    rows = []
    for anion, T, state in targets:
        p = CLS / 'event_collect' / 'event#2(Pair_breaking;survival)' / state / anion / T / 'data.txt'
        if not p.exists():
            continue
        x = cm.positive_data(p)
        xmin = float(np.min(x))
        lls, ps, l1s, l2s = [], [], [], []
        for _ in range(60):
            # randomized initial guesses for the biexponential conditional NLL
            mean_x = max(float(np.mean(x)), xmin, 1e-6)
            start = (float(rng.uniform(-3, 3)),
                     float(np.log(rng.uniform(0.2, 5) / mean_x)),
                     float(np.log(rng.uniform(0.02, 0.5) / mean_x)))
            from scipy.optimize import minimize

            def nll(theta):
                import math
                p_raw, log_l1, log_l2 = theta
                pp = 1.0 / (1.0 + math.exp(-p_raw)); l1 = math.exp(log_l1); l2 = math.exp(log_l2)
                if not (0 < pp < 1) or l1 <= 0 or l2 <= 0:
                    return 1e300
                a = math.log(pp) + math.log(l1) - l1 * x
                b = math.log1p(-pp) + math.log(l2) - l2 * x
                denom = math.log(pp * math.exp(-l1 * xmin) + (1 - pp) * math.exp(-l2 * xmin) + cm.EPS)
                ll = float(np.sum(np.maximum(a, b) + np.log(np.exp(a - np.maximum(a, b)) + np.exp(b - np.maximum(a, b))) - denom))
                return -ll if np.isfinite(ll) else 1e300
            r = minimize(nll, np.array(start), method='L-BFGS-B',
                         bounds=[(-10, 10), (-30, 5), (-30, 5)], options={'maxiter': 1000})
            if np.isfinite(r.fun):
                import math
                lls.append(-float(r.fun))
                pp = 1.0 / (1.0 + math.exp(-float(r.x[0])))
                l1 = math.exp(float(r.x[1])); l2 = math.exp(float(r.x[2]))
                fast, slow = max(l1, l2), min(l1, l2)
                ps.append(pp); l1s.append(fast); l2s.append(slow)
        lls = np.array(lls)
        best = lls.max()
        frac_at_best = float(np.mean(lls > best - 1.0))  # within 1 logL unit of best
        rows.append(dict(anion=anion, T=T, state=state, n=len(x), n_starts=len(lls),
                         frac_within_1logL=round(frac_at_best, 3),
                         logL_spread=round(float(best - lls.min()), 2),
                         fast_rate_cv=round(float(np.std(l1s) / np.mean(l1s)), 3),
                         slow_rate_cv=round(float(np.std(l2s) / np.mean(l2s)), 3)))
    with (OUT / 'parameter_stability.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == '__main__':
    print('=== CK test ===', flush=True)
    ck = ck_test()
    # summarize CK by tau
    for tau in sorted(set(r['tau'] for r in ck)):
        devs = [r['max_abs_dev'] for r in ck if r['tau'] == tau]
        print(f'  lag tau={tau} ps: max|T(k tau)-T(tau)^k| over conditions  median={np.median(devs):.3f}  max={np.max(devs):.3f}', flush=True)
    print('=== parameter stability ===', flush=True)
    for r in stability_test():
        print(f"  {r['anion']} {r['T']} {r['state']:4s} n={r['n']:5d}: frac within 1 logL of best={r['frac_within_1logL']}, "
              f"fast-rate CV={r['fast_rate_cv']}, slow-rate CV={r['slow_rate_cv']}", flush=True)
    print('WROTE', OUT)
