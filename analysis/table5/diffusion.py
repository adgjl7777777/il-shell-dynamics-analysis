"""
Compute diffusion coefficients (D) for soft, hard, and total Li+ states (Table 5).

D_soft is inferred from D_total and D_hard using the soft-state fraction θ:
    D_soft = (D_total - (1-θ) * D_hard) / θ

D_soft 오차 전파 (1차 근사):
    σ(D_soft)² = (σ_total/θ)² + ((1-θ)/θ · σ_hard)² + ((D_hard - D_soft)/θ · σ_θ)²

θ is read from results/table2_theta.csv (run analysis/table2/theta.py first).
MSD fitting ranges were manually determined by visual inspection and are hardcoded.

Output: ../../results/table5_diffusion.csv
  columns: anion, T, theta, theta_se,
           D_soft_A2ps, D_soft_se,
           D_hard_A2ps, D_hard_se,
           D_total_A2ps, D_total_se

Usage:
    python diffusion.py
"""
import os, sys
import numpy as np
import csv
from scipy.stats import linregress

CODE_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(CODE_ROOT, "results")
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, ANIONS, TEMPERATURES

# ── MSD linear-fit ranges (ps) determined by visual inspection ──────────────
# Format: {anion: {temperature: (start_ps, end_ps)}}
HARD_RANGES = {
    "fsi":  {298: (1300, 2000),  353: (350, 600),   373: (400, 600),   423: (150, 370)},
    "tfsi": {298: (1500, 5000),  353: (670, 1000),  373: (430, 600),   423: (300, 750)},
    "beti": {298: (1000, 10000), 353: (979, 1100),  373: (700, 1000),  423: (520, 750)},
}
TOTAL_RANGES = {
    "fsi":  {298: (15000, 20000), 353: (1000, 10000), 373: (1000, 10000), 423: (300, 7000)},
    "tfsi": {298: (7000, 15000),  353: (1500, 20000), 373: (700, 10000),  423: (300, 10000)},
    "beti": {298: (20000, 40000), 353: (1700, 10000), 373: (3000, 40000), 423: (600, 15000)},
}


def calculate_diffusion_coefficient(msd_data, start, end):
    """Fit MSD[start:end] ~ 6*D*t via linear regression. Returns D, D_se."""
    idx    = np.arange(start, end + 1)
    msd    = msd_data[start:end + 1]
    result = linregress(idx, msd)
    return result.slope / 6, result.stderr / 6


# ── Load θ values from table2_theta.csv ─────────────────────────────────────
theta_path = os.path.join(RESULTS_DIR, "table2_theta.csv")
if not os.path.exists(theta_path):
    raise FileNotFoundError(
        f"{theta_path} not found — run analysis/table2/theta.py first")

soft_ratios = {}
theta_ses   = {}
with open(theta_path) as fh:
    for row in csv.DictReader(fh):
        a = row["anion"]
        T = int(row["T"])
        if a not in soft_ratios:
            soft_ratios[a] = {}
            theta_ses[a]   = {}
        soft_ratios[a][T] = float(row["theta"])
        theta_ses[a][T]   = float(row.get("theta_se", 0))

# ── Compute and write results ────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "table5_diffusion.csv")
with open(OUT_CSV, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["anion", "T", "theta", "theta_se",
                     "D_soft_A2ps", "D_soft_se",
                     "D_hard_A2ps", "D_hard_se",
                     "D_total_A2ps", "D_total_se"])

    for anion in ANIONS:
        for T in TEMPERATURES:
            hard_start, hard_end   = HARD_RANGES[anion][T]
            total_start, total_end = TOTAL_RANGES[anion][T]

            hard_path  = os.path.join(DATA_ROOT, "code", "diffusion", "msd", anion, "hard", str(T), "1.0.txt")
            total_path = os.path.join(DATA_ROOT, "code", "diffusion", "msd", anion, "total", f"{T}.txt")

            hard_msd  = np.loadtxt(hard_path)
            total_msd = np.loadtxt(total_path)

            D_hard,  D_hard_se  = calculate_diffusion_coefficient(hard_msd,  hard_start,  hard_end)
            D_total, D_total_se = calculate_diffusion_coefficient(total_msd, total_start, total_end)

            theta    = soft_ratios[anion][T]
            theta_se = theta_ses[anion][T]
            D_soft   = (D_total - (1 - theta) * D_hard) / theta

            # 1차 오차 전파: σ(D_soft)² = (σ_total/θ)² + ((1-θ)/θ·σ_hard)² + ((D_hard-D_soft)/θ·σ_θ)²
            D_soft_se = np.sqrt(
                (D_total_se / theta) ** 2
                + ((1 - theta) / theta * D_hard_se) ** 2
                + ((D_hard - D_soft) / theta * theta_se) ** 2
            )

            writer.writerow([anion, T,
                             f"{theta:.6f}", f"{theta_se:.6f}",
                             f"{D_soft:.6e}", f"{D_soft_se:.2e}",
                             f"{D_hard:.6e}", f"{D_hard_se:.2e}",
                             f"{D_total:.6e}", f"{D_total_se:.2e}"])
            print(f"{anion.upper():4s}  {T}K  θ={theta:.3f}±{theta_se:.4f}  "
                  f"D_soft={D_soft:.3e}±{D_soft_se:.1e}  "
                  f"D_hard={D_hard:.3e}±{D_hard_se:.1e}  "
                  f"D_total={D_total:.3e}±{D_total_se:.1e}")

print(f"\nWritten: {OUT_CSV}")
