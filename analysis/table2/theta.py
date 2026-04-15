"""
Compute soft-state time fraction θ for each (anion, T) (→ Table 2, θ column).

Reads classification/result/{anion}/{soft|hard}/{T}/1.0_{i}.txt and sums
the total timesteps spent in each state across all 5 Li atoms.

θ의 오차는 Li 원자 5개 각각의 θ_i로부터 표준오차(SE)를 계산합니다:
    SE(θ) = std(θ_i) / sqrt(5)

Output: ../../results/table2_theta.csv
  columns: anion, T, theta, theta_se, soft_ts, hard_ts

Usage:
    python theta.py
"""
import os, sys
import numpy as np
import csv

CODE_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_DIR = os.path.join(CODE_ROOT, "classification", "result")
RESULTS_DIR  = os.path.join(CODE_ROOT, "results")
sys.path.insert(0, CODE_ROOT)
from config import ANIONS, TEMPERATURES

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = os.path.join(RESULTS_DIR, "table2_theta.csv")

rows = []
for anion in ANIONS:
    for T in TEMPERATURES:
        T_str = str(T)
        soft_total = 0
        hard_total = 0
        theta_per_atom = []

        for i in range(5):
            soft_i = 0
            hard_i = 0
            for state in ["soft", "hard"]:
                path = os.path.join(CLASSIFY_DIR, anion, state, T_str, f"1.0_{i}.txt")
                with open(path) as fh:
                    for line in fh:
                        tokens = [v for v in line.strip().split() if v.isdigit()]
                        if len(tokens) >= 2:
                            duration = int(tokens[-1]) - int(tokens[0])
                            if state == "soft":
                                soft_i += duration
                            else:
                                hard_i += duration
            total_i = soft_i + hard_i
            if total_i > 0:
                theta_per_atom.append(soft_i / total_i)
            soft_total += soft_i
            hard_total += hard_i

        total = soft_total + hard_total
        theta    = soft_total / total if total > 0 else 0.0
        theta_se = np.std(theta_per_atom, ddof=1) / np.sqrt(len(theta_per_atom)) if len(theta_per_atom) > 1 else 0.0

        rows.append({"anion": anion, "T": T,
                     "theta": round(theta, 6), "theta_se": round(theta_se, 6),
                     "soft_ts": soft_total, "hard_ts": hard_total})
        print(f"{anion:4s}  {T}K  theta={theta:.4f} ± {theta_se:.4f}")

with open(OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["anion", "T", "theta", "theta_se", "soft_ts", "hard_ts"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWritten: {OUT}")
