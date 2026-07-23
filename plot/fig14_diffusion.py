"""
Figure 14: D_soft (blue), D_hard (red), D_total (black) vs temperature.
Style: color by state; fsi=solid+o, tfsi=dashed+^, beti=dotted+s
Reads results/table5_diffusion.csv → Images/coefficient/{d_soft,d_hard,d_total}.pdf
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(CODE_ROOT, "results")
FIGURE_ROOT = os.environ.get("IL_FIGURE_ROOT", os.path.join(os.path.dirname(CODE_ROOT), "paper", "Images"))
IMAGES_DIR = os.path.join(FIGURE_ROOT, "coefficient")
os.makedirs(IMAGES_DIR, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS  = [298, 353, 373, 423]
CONV   = 1e-8   # Å²/ps → m²/s

ASTYLE = {
    "fsi":  ("-",  "o"),
    "tfsi": ("--", "^"),
    "beti": (":",  "s"),
}
ALABEL = {"fsi": "FSI", "tfsi": "TFSI", "beti": "BETI"}

STATE_COLOR = {"soft": "blue", "hard": "red", "total": "gray"}
STATE_LABEL = {"soft": "Soft", "hard": "Hard", "total": "Total"}

D    = {k: {a: {} for a in ANIONS} for k in ["soft", "hard", "total"]}
D_se = {k: {a: {} for a in ANIONS} for k in ["soft", "hard", "total"]}

with open(os.path.join(RESULTS, "table5_diffusion.csv")) as fh:
    for row in csv.DictReader(fh):
        a, T = row["anion"], int(row["T"])
        D["soft"][a][T]     = float(row["D_soft_A2ps"])  * CONV
        D_se["soft"][a][T]  = float(row["D_soft_se"])    * CONV
        D["hard"][a][T]     = float(row["D_hard_A2ps"])  * CONV
        D_se["hard"][a][T]  = float(row["D_hard_se"])    * CONV
        D["total"][a][T]    = float(row["D_total_A2ps"]) * CONV
        D_se["total"][a][T] = float(row["D_total_se"])   * CONV

for state in ["soft", "hard", "total"]:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    color = STATE_COLOR[state]

    for a in ANIONS:
        ls, mk = ASTYLE[a]
        ys  = [D[state][a].get(T, np.nan)    for T in TEMPS]
        ses = [D_se[state][a].get(T, 0)      for T in TEMPS]
        ax.errorbar(TEMPS, ys, yerr=ses,
                    color=color, linestyle=ls, marker=mk,
                    label=ALABEL[a],
                    capsize=2, markersize=4, linewidth=1.0)

    ax.set_xlabel("Temperature (K)", fontsize=8)
    ax.set_ylabel(r"Diffusion Coefficient ($\mathrm{m^2/s}$)", fontsize=7)
    ax.set_title(STATE_LABEL[state], fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(TEMPS)
    ax.tick_params(labelsize=7, direction="in")
    
    if state == "soft":
        ax.legend(fontsize=6, loc="lower right", handlelength=4.5)
        
    fig.tight_layout()

    out = os.path.join(IMAGES_DIR, f"d_{state}.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
