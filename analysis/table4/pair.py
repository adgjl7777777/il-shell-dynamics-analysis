"""
Collect intra-state Li-anion pair survival times f(n) for burstiness analysis.

Loads x11_loyal (soft) and x22_loyal (hard) trajectory files from classification/x/
and records pair survival durations. For the hard state, TABLE2_NSTART[anion][T] is
subtracted from each duration (matching the power-law fit n_start offset in Table 2).

Output: classification/event_collect/event#2(Pair_breaking;survival)/{soft|hard}/{anion}/{T}/data.txt

Usage:
    python pair.py <anion>
"""
import numpy as np
import os, sys

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")
sys.path.insert(0, CODE_ROOT)
from config import TABLE2_NSTART

anion = sys.argv[1]
TRAIL = 1.0   # canonical threshold

for T in ["298", "353", "373", "423"]:
    realhard, realsoft = [], []
    nstart = TABLE2_NSTART[anion][int(T)]
    for i in range(5):
        hard_path = os.path.join(CLASSIFY_DIR, "x", anion, T, "x22_loyal", f"{TRAIL}_{i}.txt")
        soft_path = os.path.join(CLASSIFY_DIR, "x", anion, T, "x11_loyal", f"{TRAIL}_{i}.txt")
        hard = np.loadtxt(hard_path)
        soft = np.loadtxt(soft_path)
        if hard.ndim == 2:
            realhard.extend([j[-1] - j[0] - nstart for j in hard if j[-1] - j[0] - nstart > 0])
        if soft.ndim == 2:
            realsoft.extend([j[-1] - j[0] for j in soft])

    base = os.path.join(CLASSIFY_DIR, "event_collect", "event#2(Pair_breaking;survival)")
    for state, data in [("hard", realhard), ("soft", realsoft)]:
        out_dir = os.path.join(base, state, anion, T)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, "data.txt"), data)

    print(f"{anion} {T}K: soft N={len(realsoft)}, hard N={len(realhard)}")
