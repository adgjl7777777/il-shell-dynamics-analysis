"""
Collect total (state-agnostic) shell-exchange inter-event times for x burstiness.

Reads shell_exchange.txt for all anions/temperatures and computes the inter-event
time series across all Li atoms without soft/hard separation.
Used to compute the total burstiness A(x) reported in Table 4 of the paper.

Output: classification/event_collect/event#1(Shell_change;interevent)/total/{anion}/{T}/total.txt

Usage:
    python total.py
"""
import sys, os

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, ANIONS, TEMPERATURES

import numpy as np

for anion in ANIONS:
    for T in TEMPERATURES:
        T = str(T)
        se_path = os.path.join(DATA_ROOT, anion, T, "shell_exchange.txt")
        se = np.loadtxt(se_path, int)
        Nsteps, Natoms = np.shape(se)

        prev     = np.zeros(Natoms, int)
        interval = [[] for _ in range(Natoms - 1)]
        for t in range(min(100000, Nsteps)):
            for i in range(Natoms - 1):
                if se[t, i + 1] == 1:
                    if prev[i] != 0:
                        interval[i].append(t - prev[i])
                    prev[i] = t

        interval_total = np.array([v for sub in interval for v in sub])

        base = os.path.join(CLASSIFY_DIR, "event_collect", "event#1(Shell_change;interevent)")
        out_dir = os.path.join(base, "total", anion, T)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, "total.txt"), interval_total)
        print(f"{anion} {T}K: total inter-event N={len(interval_total)}")
