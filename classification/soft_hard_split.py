"""
Classify Li-anion pair trajectories into x11/x12/x21/x22/loyal categories.

x11 = soft→soft (intra-soft), x22 = hard→hard (intra-hard)
x12 = soft→hard, x21 = hard→soft (inter-state transitions)
x11_loyal / x22_loyal = pairs that stay within one state throughout

Usage:
    python soft_hard_split.py <anion>
"""
import numpy as np
import os, sys
import time
from tqdm import trange

anion = sys.argv[1]
TRAIL = 1.0   # canonical classification threshold (multi=1.0)

for T in ["298", "353", "373", "423"]:
    for cat in ["x11", "x12", "x21", "x22", "x11_loyal", "x22_loyal"]:
        os.makedirs(f"x/{anion}/{T}/{cat}", exist_ok=True)

    for i in range(5):
        se = np.loadtxt(f"pair/{anion}/{T}/{i}.txt")
        x11, x12, x21, x22, x11_loyal, x22_loyal = [], [], [], [], [], []

        with open(f"result/{anion}/hard/{T}/{TRAIL}_{i}.txt") as f:
            hard = [[int(j) for j in line.strip().split() if j.isdigit()] for line in f]
        with open(f"result/{anion}/soft/{T}/{TRAIL}_{i}.txt") as f:
            soft = [[int(j) for j in line.strip().split() if j.isdigit()] for line in f]

        hard = [[j[0], j[-1], 1] for j in hard]
        soft = [[j[0], j[-1], 0] for j in soft]
        tots = sorted(hard + soft)
        tots.append([10000000, 10000000, 2])

        index_i = 0
        index_f = 0
        for event in se:
            if len(event) > 1:
                while event[0] > tots[index_i][0]:
                    index_i += 1
                while event[1] > tots[index_f][0]:
                    index_f += 1
                index_i -= 1
                index_f -= 1
                if event[0] == tots[index_i][1]:
                    index_i += 1
                if event[1] == tots[index_f][0]:
                    index_f -= 1

                if index_i == index_f:
                    if tots[index_i][-1] == 0:
                        x11_loyal.append(event)
                        x11.append(event)
                    else:
                        x22_loyal.append(event)
                        x22.append(event)
                else:
                    si, sf = tots[index_i][-1], tots[index_f][-1]
                    if si == 0 and sf == 0:
                        x11.append(event)
                    elif si == 0 and sf == 1:
                        x12.append(event)
                    elif si == 1 and sf == 0:
                        x21.append(event)
                    else:
                        x22.append(event)

        np.savetxt(f"x/{anion}/{T}/x11/{TRAIL}_{i}.txt",       x11)
        np.savetxt(f"x/{anion}/{T}/x12/{TRAIL}_{i}.txt",       x12)
        np.savetxt(f"x/{anion}/{T}/x21/{TRAIL}_{i}.txt",       x21)
        np.savetxt(f"x/{anion}/{T}/x22/{TRAIL}_{i}.txt",       x22)
        np.savetxt(f"x/{anion}/{T}/x11_loyal/{TRAIL}_{i}.txt", x11_loyal)
        np.savetxt(f"x/{anion}/{T}/x22_loyal/{TRAIL}_{i}.txt", x22_loyal)
