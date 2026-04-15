"""
Figure 13: MSD plot of fsi 298K — soft (blue), hard (blue), total (black).
Saves Images/Diffusion/fsi/298/{soft,hard,total}.pdf

style: soft=blue, hard=red, total=black (log-log axes)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, FIGURE_ROOT

IMAGES_DIR = os.path.join(FIGURE_ROOT, "Diffusion", "fsi", "298")
os.makedirs(IMAGES_DIR, exist_ok=True)

HARD_RANGE  = (1300, 2000)   # ps
TOTAL_RANGE = (15000, 20000) # ps

soft_path  = os.path.join(DATA_ROOT, "code", "diffusion", "msd", "fsi", "soft",  "298", "1.0.txt")
hard_path  = os.path.join(DATA_ROOT, "code", "diffusion", "msd", "fsi", "hard",  "298", "1.0.txt")
total_path = os.path.join(DATA_ROOT, "code", "diffusion", "msd", "fsi", "total", "298.txt")

soft_msd  = np.loadtxt(soft_path)
hard_msd  = np.loadtxt(hard_path)
total_msd = np.loadtxt(total_path)

def plot_msd(msd, color, title, fname, fit_range=None):
    msd = msd[1:]   # skip t=0 (MSD[0] is undefined/near-zero)
    n = len(msd)
    t = np.arange(1, n + 1)   # ps

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(t, np.abs(msd), color=color, linewidth=0.8, label="MSD")

    pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$ (ps)", fontsize=8)
    ax.set_ylabel(r"MSD ($\AA^2$)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7, direction="in")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, fname)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

plot_msd(soft_msd,  "blue",  "FSI, 298K, Soft",  "soft.pdf")
plot_msd(hard_msd,  "red",   "FSI, 298K, Hard",  "hard.pdf")
plot_msd(total_msd, "black", "FSI, 298K, Total", "total.pdf")
