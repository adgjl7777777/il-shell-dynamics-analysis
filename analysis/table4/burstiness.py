"""
Compute finite-size-corrected burstiness parameter A_N for each state (Table 4).

A_N = (sqrt(N+1)*r - sqrt(N-1)) / ((sqrt(N+1)-2)*r + sqrt(N-1))
where r = std/mean of the inter-event (or survival) times.

Bootstrap (N_BOOT=1000) gives 95% CI for A_N.

Event types computed:
  soft_hard_duration              → A(h_soft), A(h_hard)  — state residence times
  event#2(Pair_breaking;survival) → A(f_soft), A(f_hard)  — intra-state pair survival
  event#1(Shell_change;interevent)→ A(x_soft), A(x_hard)  — shell-change inter-event
  event#1(Shell_change;interevent)/total → A(x_total)      — state-agnostic

Input:  classification/event_collect/{event_type}/{state}/{anion}/{T}/data.txt
Output: ../../results/table4_burstiness.csv

Usage:
    python burstiness.py <anion>
"""
import os, sys
import numpy as np
import csv

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")
RESULTS_DIR  = os.path.join(CODE_ROOT, "results")
sys.path.insert(0, CODE_ROOT)
from config import TEMPERATURES

ANION  = sys.argv[1]
N_BOOT = 1000
RNG    = np.random.default_rng(42)

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "table4_burstiness.csv")
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["anion", "T", "event_type", "state", "A_N", "A_lo_95", "A_hi_95"])


def burstiness_A(data):
    n    = len(data)
    mean = data.mean()
    std  = data.std()
    if mean == 0:
        return np.nan
    r = std / mean
    num = np.sqrt(n + 1) * r - np.sqrt(n - 1)
    den = (np.sqrt(n + 1) - 2) * r + np.sqrt(n - 1)
    return num / den if den != 0 else np.nan


def bootstrap_ci(data, n_boot=N_BOOT, ci=0.95):
    boots = [burstiness_A(RNG.choice(data, size=len(data), replace=True))
             for _ in range(n_boot)]
    boots = np.array([b for b in boots if not np.isnan(b)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi


SPLIT_TYPES = [
    "event#1(Shell_change;interevent)",
    "event#2(Pair_breaking;survival)",
    "soft_hard_duration",
]

with open(OUT_CSV, "a", newline="") as fh:
    writer = csv.writer(fh)

    for Atype in SPLIT_TYPES:
        for T in TEMPERATURES:
            for sh in ["soft", "hard"]:
                path = os.path.join(CLASSIFY_DIR, "event_collect", Atype, sh, ANION, str(T), "data.txt")
                mydata = np.loadtxt(path)
                A          = burstiness_A(mydata)
                A_lo, A_hi = bootstrap_ci(mydata)
                writer.writerow([ANION, T, Atype, sh, f"{A:.6f}", f"{A_lo:.6f}", f"{A_hi:.6f}"])
                print(f"A({Atype}/{sh}) {ANION} {T}K = {A:.3f} [{A_lo:.3f}, {A_hi:.3f}]")

    for T in TEMPERATURES:
        path = os.path.join(CLASSIFY_DIR, "event_collect",
                            "event#1(Shell_change;interevent)", "total", ANION, str(T), "total.txt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found — run total.py first")
            continue
        mydata     = np.loadtxt(path)
        A          = burstiness_A(mydata)
        A_lo, A_hi = bootstrap_ci(mydata)
        writer.writerow([ANION, T, "x_total", "total", f"{A:.6f}", f"{A_lo:.6f}", f"{A_hi:.6f}"])
        print(f"A(x_total) {ANION} {T}K = {A:.3f} [{A_lo:.3f}, {A_hi:.3f}]")
