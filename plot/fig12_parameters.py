"""
Figure 12: θ (ratio.pdf), α (a.pdf), β (b.pdf) vs temperature.
Style:
  - θ: all black/gray; fsi=solid+o, tfsi=dashed+^, beti=dotted+s
  - α, β: soft=blue, hard=red; fsi=solid+o, tfsi=dashed+^, beti=dotted+s
Reads results/table2_*.csv → Images/coefficient/{ratio,a,b}.pdf
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(CODE_ROOT, "results")
IMAGES_DIR = os.path.join(os.path.dirname(CODE_ROOT), "paper", "Images", "coefficient")
os.makedirs(IMAGES_DIR, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS  = [298, 353, 373, 423]

# Anion → (linestyle, marker, gray shade for θ plot)
ASTYLE = {
    "fsi":  ("-",  "o", "black"),
    "tfsi": ("--", "^", "gray"),
    "beti": (":",  "s", "darkgray"),
}
ALABEL = {"fsi": "FSI", "tfsi": "TFSI", "beti": "BETI"}

# ── Load data ────────────────────────────────────────────────────────
theta, theta_se = {}, {}
with open(os.path.join(RESULTS, "table2_theta.csv")) as fh:
    for row in csv.DictReader(fh):
        a, T = row["anion"], int(row["T"])
        theta.setdefault(a, {})[T]    = float(row["theta"])
        theta_se.setdefault(a, {})[T] = float(row["theta_se"])

alpha, alpha_se = {"soft":{}, "hard":{}}, {"soft":{}, "hard":{}}
with open(os.path.join(RESULTS, "table2_h_exponential.csv")) as fh:
    for row in csv.DictReader(fh):
        a, T, st = row["anion"], int(row["T"]), row["state"]
        alpha[st].setdefault(a, {})[T]    = float(row["alpha"])
        alpha_se[st].setdefault(a, {})[T] = float(row["alpha_se"])

beta, beta_se = {"soft":{}, "hard":{}}, {"soft":{}, "hard":{}}
with open(os.path.join(RESULTS, "table2_f_powerlaw.csv")) as fh:
    for row in csv.DictReader(fh):
        a, T, cat = row["anion"], int(row["T"]), row["category"]
        if cat == "x11_loyal":
            beta["soft"].setdefault(a, {})[T]    = float(row["beta"])
            beta_se["soft"].setdefault(a, {})[T] = float(row["beta_se"])
        elif cat == "x22_loyal":
            beta["hard"].setdefault(a, {})[T]    = float(row["beta"])
            beta_se["hard"].setdefault(a, {})[T] = float(row["beta_se"])

# ── ratio.pdf : θ ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for a in ANIONS:
    ls, mk, gc = ASTYLE[a]
    ys  = [theta[a].get(T, np.nan)    for T in TEMPS]
    ses = [theta_se[a].get(T, 0)      for T in TEMPS]
    ax.errorbar(TEMPS, ys, yerr=ses,
                color=gc, linestyle=ls, marker=mk,
                label=ALABEL[a], capsize=2, markersize=4, linewidth=1.0)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"Soft ratio $\theta$", fontsize=8)
ax.set_xticks(TEMPS)
ax.tick_params(labelsize=7, direction="in")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "ratio.pdf"), dpi=300)
plt.close(fig)
print("Saved: ratio.pdf")

# ── a.pdf : α (soft=blue, hard=red) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for a in ANIONS:
    ls, mk, _ = ASTYLE[a]
    # soft (blue, filled marker)
    ys_s  = [alpha["soft"][a].get(T, np.nan) for T in TEMPS]
    ses_s = [alpha_se["soft"][a].get(T, 0)   for T in TEMPS]
    ax.errorbar(TEMPS, ys_s, yerr=ses_s,
                color="blue", linestyle=ls, marker=mk,
                label=f"{ALABEL[a]} soft", capsize=2, markersize=4, linewidth=1.0)
    # hard (red, same linestyle/marker)
    ys_h  = [alpha["hard"][a].get(T, np.nan) for T in TEMPS]
    ses_h = [alpha_se["hard"][a].get(T, 0)   for T in TEMPS]
    ax.errorbar(TEMPS, ys_h, yerr=ses_h,
                color="red", linestyle=ls, marker=mk,
                label=f"{ALABEL[a]} hard", capsize=2, markersize=4, linewidth=1.0,
                alpha=0.7)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"$\alpha$", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(TEMPS)
ax.tick_params(labelsize=7, direction="in")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "a.pdf"), dpi=300)
plt.close(fig)
print("Saved: a.pdf")

# ── b.pdf : β (soft=blue, hard=red) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for a in ANIONS:
    ls, mk, _ = ASTYLE[a]
    ys_s  = [beta["soft"][a].get(T, np.nan) for T in TEMPS]
    ses_s = [beta_se["soft"][a].get(T, 0)   for T in TEMPS]
    ax.errorbar(TEMPS, ys_s, yerr=ses_s,
                color="blue", linestyle=ls, marker=mk,
                label=f"{ALABEL[a]} soft", capsize=2, markersize=4, linewidth=1.0)
    ys_h  = [beta["hard"][a].get(T, np.nan) for T in TEMPS]
    ses_h = [beta_se["hard"][a].get(T, 0)   for T in TEMPS]
    ax.errorbar(TEMPS, ys_h, yerr=ses_h,
                color="red", linestyle=ls, marker=mk,
                label=f"{ALABEL[a]} hard", capsize=2, markersize=4, linewidth=1.0,
                alpha=0.7)
ax.set_xlabel("Temperature (K)", fontsize=8)
ax.set_ylabel(r"$\beta$", fontsize=8)
ax.set_xticks(TEMPS)
ax.tick_params(labelsize=7, direction="in")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(IMAGES_DIR, "b.pdf"), dpi=300)
plt.close(fig)
print("Saved: b.pdf")
