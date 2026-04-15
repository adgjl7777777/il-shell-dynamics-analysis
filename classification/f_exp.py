"""
Fit exponential h(n) distributions to x11/x12/x21/x22/x11_loyal/x22_loyal
state-duration data and write fit parameters (α ± SE) to yame2.txt (→ Table 2).

Usage:
    python f_exp.py <anion>
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import dist

anion = sys.argv[1]
TRAIL = 1.0   # canonical threshold

color = ["#5555FF", "#55FF55", "#FFAA55", "#FF5555"]
namer = ["x11", "x12", "x21", "x22", "x11_loyal", "x22_loyal"]

tx = {cat: {T: [] for T in ["298", "353", "373", "423"]}
      for cat in namer}

for T in ["298", "353", "373", "423"]:
    for cat in namer:
        os.makedirs(f"plot/{anion}/{cat}", exist_ok=True)
    for i in range(5):
        for cat in namer:
            data = np.loadtxt(f"x/{anion}/{T}/{cat}/{TRAIL}_{i}.txt")
            if data.ndim == 2:
                tx[cat][T] += list(data.reshape(-1, 2))

for ind, cat in enumerate(namer):
    int_dis, x = {}, {}
    for T in ["298", "353", "373", "423"]:
        durations = [row[1] - row[0] for row in tx[cat][T]]
        x[T], int_dis[T], a, a_se, b, b_se, c = dist.exp_calc(durations)

        os.makedirs(f"hist/{anion}/{cat}", exist_ok=True)
        with open(f"hist/{anion}/{cat}/{T}_exp.txt", "w") as f:
            for xi, yi in zip(x[T], int_dis[T]):
                f.write(f"{xi}, {yi}\n")
        with open("yame2.txt", "a") as f:
            f.write(f"{anion},{T},{cat},{a},{a_se}\n")

    maxer = [2000, 0, 0, 0]
    for i, T in enumerate(["298", "353", "373", "423"]):
        plt.plot(x[T], int_dis[T], color=color[i], label=f"{T}K")
        if x[T]:
            maxer[i] = max(x[T])
    plt.yscale("log", base=10)
    plt.xlim([1, max(maxer)])
    plt.ylim([1e-5, 1])
    plt.xlabel(r"$\tau$ (ps)")
    plt.ylabel(r"$P(\tau)$")
    plt.legend()
    plt.savefig(f"plot/{anion}/{cat}/exp.png", dpi=600)
    plt.clf()
    plt.cla()
    plt.close()
