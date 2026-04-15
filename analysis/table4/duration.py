"""
Compute soft/hard state residence durations h(n) from classification result files.

Reads classification/result/{anion}/{soft|hard}/{T}/1.0_{i}.txt and records
the duration (end - start - CUTOFF) of each state segment. Excludes segments
touching t=0 (boundary artifact) and durations ≤ CUTOFF (inertial rattling < 25 ps).

Output: classification/event_collect/soft_hard_duration/{soft|hard}/{anion}/{T}/data.txt

Usage:
    python duration.py <anion>
"""
import numpy as np
import os, sys

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")

anion  = sys.argv[1]
TRAIL  = 1.0   # canonical threshold
CUTOFF = 50    # subtract 50 ts from each duration (removes inertial rattling <25 ps)

for T in ["298", "353", "373", "423"]:
    realhard, realsoft = [], []
    for i in range(5):
        for state, container in [("hard", realhard), ("soft", realsoft)]:
            path = os.path.join(CLASSIFY_DIR, "result", anion, state, T, f"{TRAIL}_{i}.txt")
            with open(path) as f:
                segs = [[int(j) for j in line.strip().split() if j.isdigit()] for line in f]
            container.extend([j[-1] - j[0] - CUTOFF for j in segs
                               if j[0] != 0 and j[-1] - j[0] - CUTOFF > 0])

    for state, data in [("hard", realhard), ("soft", realsoft)]:
        out_dir = os.path.join(CLASSIFY_DIR, "event_collect", "soft_hard_duration", state, anion, T)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, "data.txt"), data)

    print(f"{anion} {T}K: soft N={len(realsoft)}, hard N={len(realhard)}")
