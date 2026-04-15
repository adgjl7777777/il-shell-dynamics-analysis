"""
Figure 11: Burstiness parameter A_N vs temperature, one subplot per anion.
Style: soft=blue, hard=red, x_total=black; solid=soft, dashed=hard.
Reads results/table4_burstiness.csv → Images/A/{anion}_plot.pdf
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(CODE_ROOT, "results")
IMAGES_DIR = os.path.join(os.path.dirname(CODE_ROOT), "paper", "Images", "A")
os.makedirs(IMAGES_DIR, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
LABELS = {"fsi": "FSI", "tfsi": "TFSI", "beti": "BETI"}
TEMPS  = [298, 353, 373, 423]

# (event_type, state) → (label, color, linestyle, marker)
SERIES = [
    ("soft_hard_duration",              "soft",  r"$h_\mathrm{soft}$", "blue",  "-",  "o"),
    ("soft_hard_duration",              "hard",  r"$h_\mathrm{hard}$", "blue",  "--", "o"),
    ("x_total",                         "total", r"$x$",               "black", "-",  "s"),
    ("event#2(Pair_breaking;survival)", "soft",  r"$f_\mathrm{soft}$", "red",   "-",  "^"),
    ("event#2(Pair_breaking;survival)", "hard",  r"$f_\mathrm{hard}$", "red",   "--", "^"),
]

# Load data
data = {a: {T: {} for T in TEMPS} for a in ANIONS}
with open(os.path.join(RESULTS, "table4_burstiness.csv")) as fh:
    for row in csv.DictReader(fh):
        a, T = row["anion"], int(row["T"])
        key  = (row["event_type"], row["state"])
        if a in data and T in data[a]:
            data[a][T][key] = (float(row["A_N"]),
                               float(row["A_lo_95"]),
                               float(row["A_hi_95"]))

for anion in ANIONS:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for (etype, state, label, color, ls, mk) in SERIES:
        key = (etype, state)
        ys, lo_e, hi_e = [], [], []
        for T in TEMPS:
            val = data[anion][T].get(key)
            if val is None:
                ys.append(np.nan); lo_e.append(0); hi_e.append(0)
            else:
                A, lo, hi = val
                ys.append(A)
                lo_e.append(A - lo)
                hi_e.append(hi - A)

        # lighter blue for h_hard
        plot_color = "cornflowerblue" if (color == "blue" and state == "hard") else \
                     "lightcoral"     if (color == "red"  and state == "hard") else color

        ax.errorbar(TEMPS, ys, yerr=[lo_e, hi_e],
                    color=plot_color, linestyle=ls, marker=mk,
                    label=label, capsize=2, markersize=4, linewidth=1.0)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Temperature (K)", fontsize=8)
    ax.set_ylabel(r"$A_N$", fontsize=8)
    ax.set_title(LABELS[anion], fontsize=9)
    ax.set_xticks(TEMPS)
    ax.tick_params(labelsize=7, direction="in")
    ax.legend(fontsize=6, loc="best", framealpha=0.7)
    fig.tight_layout()

    out = os.path.join(IMAGES_DIR, f"{anion}_plot.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
