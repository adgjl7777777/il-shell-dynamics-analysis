"""
Fit exponential decay to soft/hard state residence time distributions h(n) (→ Table 2).

Reads duration data from classification/event_collect/soft_hard_duration/{soft|hard}/{anion}/{T}/data.txt
(produced by analysis/table4/duration.py) and writes fit parameters (α ± SE).

Uses fixed 100-bin histogram without x-shifting (h_calc in dist.py),
matching the approach that produced the paper's Table 2 α values.

Output: ../../results/table2_h_exponential.csv

Usage:
    python h_fit.py <anion>
"""
import numpy as np
import os, sys

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")
RESULTS_DIR  = os.path.join(CODE_ROOT, "results")
sys.path.insert(0, os.path.join(CODE_ROOT, "classification"))
import dist

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "table2_h_exponential.csv")
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w") as f:
        f.write("anion,T,state,alpha,alpha_se\n")

anion = sys.argv[1]

for T in ["298", "353", "373", "423"]:
    for state in ["soft", "hard"]:
        path = os.path.join(CLASSIFY_DIR, "event_collect", "soft_hard_duration",
                            state, anion, T, "data.txt")
        data = np.loadtxt(path)
        _, _, a, a_se, b, b_se, _ = dist.h_calc(list(data))
        with open(OUT_CSV, "a") as f:
            f.write(f"{anion},{T},{state},{a},{a_se}\n")
        print(f"alpha(h_{state}) {anion} {T}K = {a:.4e} +/- {a_se:.4e}")
