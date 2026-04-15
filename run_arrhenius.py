"""
Arrhenius analysis (B3-1, B3-2).

B3-1: ln(alpha) vs 1/T → activation energy E_a for soft/hard state transitions
B3-2: ln(theta/(1-theta)) vs 1/T → free energy difference ΔG

Style follows fig12_parameters.py:
  alpha panels: soft=blue, hard=red; fsi=solid+o, tfsi=dashed+^, beti=dotted+s
  theta panel:  black/gray tones;    fsi=solid+o, tfsi=dashed+^, beti=dotted+s

Output:
  results/arrhenius.csv
  paper/Images/coefficient/arrhenius.pdf   (matches fig12 output directory)

Usage:
    cd path/to/il_paper/code
    python run_arrhenius.py
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

CODE_ROOT   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_ROOT, "results")
FIG_DIR     = os.path.join(CODE_ROOT, "..", "paper", "Images", "coefficient")

sys.path.insert(0, CODE_ROOT)
from config import ANIONS, TEMPERATURES

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

k_B  = 8.617333e-5   # eV/K
TEMPS = np.array(TEMPERATURES, dtype=float)

# ── Style matching fig12_parameters.py ───────────────────────────────────────
ASTYLE = {
    "fsi":  ("-",  "o", "black"),
    "tfsi": ("--", "^", "gray"),
    "beti": (":",  "s", "darkgray"),
}
ALABEL = {"fsi": "FSI$^-$", "tfsi": "TFSI$^-$", "beti": "BETI$^-$"}

# ── Load data ─────────────────────────────────────────────────────────────────
alpha_data = {}
with open(os.path.join(RESULTS_DIR, "table2_h_exponential.csv")) as fh:
    for row in csv.DictReader(fh):
        alpha_data[(row["anion"], int(row["T"]), row["state"])] = (
            float(row["alpha"]), float(row["alpha_se"]))

theta_data = {}
with open(os.path.join(RESULTS_DIR, "table2_theta.csv")) as fh:
    for row in csv.DictReader(fh):
        theta_data[(row["anion"], int(row["T"]))] = (
            float(row["theta"]), float(row["theta_se"]))


def arrhenius_fit(T_vals, y_vals):
    ln_y = np.log(y_vals)
    inv_T = 1.0 / T_vals
    res   = linregress(inv_T, ln_y)
    E_a   = -res.slope * k_B
    E_a_se = res.stderr * k_B
    return E_a, E_a_se, res.rvalue**2, res.slope, res.intercept


# ── Compute results and build figure ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
ax_soft, ax_hard, ax_dg = axes

rows = []
inv_T_vals = 1000.0 / TEMPS   # 10³/T for x-axis

for anion in ANIONS:
    ls, mk, gc = ASTYLE[anion]
    lbl = ALABEL[anion]

    a_soft   = np.array([alpha_data[(anion, T, "soft")][0]  for T in TEMPERATURES])
    a_soft_e = np.array([alpha_data[(anion, T, "soft")][1]  for T in TEMPERATURES])
    a_hard   = np.array([alpha_data[(anion, T, "hard")][0]  for T in TEMPERATURES])
    a_hard_e = np.array([alpha_data[(anion, T, "hard")][1]  for T in TEMPERATURES])
    theta    = np.array([theta_data[(anion, T)][0]          for T in TEMPERATURES])
    theta_e  = np.array([theta_data[(anion, T)][1]          for T in TEMPERATURES])

    Ea_s, Ea_s_se, r2_s, sl_s, ic_s = arrhenius_fit(TEMPS, a_soft)
    Ea_h, Ea_h_se, r2_h, sl_h, ic_h = arrhenius_fit(TEMPS, a_hard)

    log_odds   = np.log(theta / (1 - theta))
    log_odds_e = theta_e / (theta * (1 - theta))
    res_dg     = linregress(1.0 / TEMPS, log_odds)
    dH         = -res_dg.slope * k_B * 1000   # meV

    # ── soft panel (blue) ─────────────────────────────────────────────────
    ax_soft.errorbar(inv_T_vals, np.log(a_soft),
                     yerr=a_soft_e / a_soft,
                     color="blue", linestyle=ls, marker=mk,
                     label=lbl, capsize=2, markersize=4, linewidth=1.0)
    x_fit = np.linspace(inv_T_vals.min(), inv_T_vals.max(), 60)
    ax_soft.plot(x_fit, sl_s * (x_fit / 1000) + ic_s,
                 color="blue", linestyle=ls, linewidth=0.8, alpha=0.6)

    # ── hard panel (red) ─────────────────────────────────────────────────
    ax_hard.errorbar(inv_T_vals, np.log(a_hard),
                     yerr=a_hard_e / a_hard,
                     color="red", linestyle=ls, marker=mk,
                     label=lbl, capsize=2, markersize=4, linewidth=1.0)
    ax_hard.plot(x_fit, sl_h * (x_fit / 1000) + ic_h,
                 color="red", linestyle=ls, linewidth=0.8, alpha=0.6)

    # ── theta panel (black/gray like ratio.pdf) ───────────────────────────
    ax_dg.errorbar(inv_T_vals, log_odds, yerr=log_odds_e,
                   color=gc, linestyle=ls, marker=mk,
                   label=lbl, capsize=2, markersize=4, linewidth=1.0)
    ax_dg.plot(x_fit, res_dg.slope * (x_fit / 1000) + res_dg.intercept,
               color=gc, linestyle=ls, linewidth=0.8, alpha=0.6)

    rows.append({
        "anion": anion,
        "Ea_soft_meV": round(Ea_s * 1000, 1), "Ea_soft_se_meV": round(Ea_s_se * 1000, 1),
        "R2_soft": round(r2_s, 4),
        "Ea_hard_meV": round(Ea_h * 1000, 1), "Ea_hard_se_meV": round(Ea_h_se * 1000, 1),
        "R2_hard": round(r2_h, 4),
        "dH_meV": round(dH, 1),
    })
    print(f"{anion:4s}  Ea(soft)={Ea_s*1000:.1f}±{Ea_s_se*1000:.1f} meV R²={r2_s:.3f}"
          f"  |  Ea(hard)={Ea_h*1000:.1f}±{Ea_h_se*1000:.1f} meV R²={r2_h:.3f}"
          f"  |  ΔH={dH:.1f} meV")

# ── Axis formatting ───────────────────────────────────────────────────────────
for ax, ylabel, title in [
    (ax_soft,
     r"$\ln\,\alpha(h_\mathrm{soft})\ [\mathrm{ps}^{-1}]$",
     r"$\alpha(h_\mathrm{soft})$  (soft state)"),
    (ax_hard,
     r"$\ln\,\alpha(h_\mathrm{hard})\ [\mathrm{ps}^{-1}]$",
     r"$\alpha(h_\mathrm{hard})$  (hard state)"),
    (ax_dg,
     r"$\ln\,[\theta/(1-\theta)]$",
     r"Soft-state equilibrium"),
]:
    ax.set_xlabel(r"$10^3/T\ \mathrm{(K^{-1})}$", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7, direction="in")
    ax.legend(fontsize=7, framealpha=0.7)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "arrhenius.pdf")
plt.savefig(out_fig, bbox_inches="tight", dpi=300)
plt.close()
print(f"\nFigure: {out_fig}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_csv = os.path.join(RESULTS_DIR, "arrhenius.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Written: {out_csv}\nDone.")
