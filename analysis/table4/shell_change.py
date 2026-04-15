"""
Collect shell-exchange inter-event times per state for burstiness analysis.

Reads classification/result/{anion}/{soft|hard}/{T}/1.0_{i}.txt and records
the time interval between consecutive shell-change events within each state.

Output: classification/event_collect/event#1(Shell_change;interevent)/{soft|hard}/{anion}/{T}/data.txt

Usage:
    python shell_change.py <anion>
"""
import numpy as np
import os, sys

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification")

anion = sys.argv[1]
TRAIL = 1.0   # canonical threshold

for T in ["298", "353", "373", "423"]:
    realhard, realsoft = [], []
    for i in range(5):
        for state, container in [("hard", realhard), ("soft", realsoft)]:
            path = os.path.join(CLASSIFY_DIR, "result", anion, state, T, f"{TRAIL}_{i}.txt")
            with open(path) as f:
                segs = [[int(j) for j in line.strip().split() if j.isdigit()] for line in f]
            for seg in segs:
                for j in range(len(seg) - 1):
                    if seg[j + 1] != seg[j] and seg[j] != 0:
                        container.append(seg[j + 1] - seg[j])

    base = os.path.join(CLASSIFY_DIR, "event_collect", "event#1(Shell_change;interevent)")
    for state, data in [("hard", realhard), ("soft", realsoft)]:
        out_dir = os.path.join(base, state, anion, T)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, "data.txt"), data)

    print(f"{anion} {T}K: soft N={len(realsoft)}, hard N={len(realhard)}")
