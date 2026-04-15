"""
Compute inter-event time distribution P(τ) for total shell-change events (Table 1).

Fits a power-law to the inter-event distribution to extract β and Δt_exp,
which are the classification threshold values in TABLE1 of config.py.

Outputs:
  inter_output/{anion}/{T}/inter_info.txt    - β, SE, Δt statistics
  inter_output/{anion}/{T}/inter_int.txt     - binned P(τ) data
  inter_output/{anion}/{T}/inter_pow.pdf     - log-log plot
  ../../results/table1_inter_event.csv       - β ± SE per (anion, T) (→ Table 1)

Usage:
    python inter.py <anion> <T>
"""
import sys, os
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "classification"))
from config import DATA_ROOT, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import dist

anion = sys.argv[1]
T     = sys.argv[2]
print(f"Calculating inter-event distribution for {anion} {T}K...")

pair_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inter_output")
os.makedirs(f"{pair_path}/{anion}/{T}", exist_ok=True)

se_path = f"{DATA_ROOT}/{anion}/{T}/shell_exchange.txt"
se = np.loadtxt(se_path, int)

Nsteps, Natoms = np.shape(se)
prev = np.zeros(Natoms, int)
interval = [[] for _ in range(Natoms-1)]
for t in range(min(100000,Nsteps)):
    if t%1000 == 0:
        print(f"Calculating tau: {t}/{Nsteps} steps")
    for i in range(Natoms-1):
        if se[t,i+1] == 1:
            if prev[i] != 0:
                interval[i].append(t-prev[i])
            prev[i]=t
interval_total=[]
for i in interval:
    interval_total.extend(i)
sur=np.array(interval_total)

x,int_dis,coeff,coeff_se,inter,inter_se,indic = dist.pow_calc(sur)
with open(f"{pair_path}/{anion}/{T}/inter_info.txt","w") as f:
    f.write("Average:\n")
    f.write(str(np.average(sur)))
    f.write("\n")
    f.write("Median:\n")
    f.write(str(np.median(sur)))
    f.write("\n")
    f.write("95%:\n")
    f.write(str(np.quantile(sur,0.95)))
    f.write("\n")
    f.write("99%:\n")
    f.write(str(np.quantile(sur,0.99)))
    f.write("\n")
    f.write("Coeff:\n")
    f.write(str(coeff))
    f.write("\n")
    f.write("Coeff_SE:\n")
    f.write(str(coeff_se))
    f.write("\n")
    f.write("Intercept:\n")
    f.write(str(inter))
    f.write("\n")
    f.write("Intercept_SE:\n")
    f.write(str(inter_se))
    f.write("\n")
    f.write("Indicator:\n")
    f.write(str(indic))
    f.write("\n")

out = open(f"{pair_path}/{anion}/{T}/inter_int.txt", 'w')
for i in range(len(x)):
    out.write(f"{x[i]:.8f}    {int_dis[i]:.8f}\n")
out.close()

plt.plot(x,int_dis,color='black')
plt.yscale("log",base=10)
plt.xscale("log",base=10)
plt.xlim([1,100000])
plt.ylim([10**(-8),1])
plt.xlabel("$\\tau(ps)$")
plt.ylabel("$P(\\tau)$")
plt.title(f"Inter-event Time Distribution at {T}K")
plt.savefig(f"{pair_path}/{anion}/{T}/inter_pow.pdf",dpi=600)
plt.cla(); plt.clf(); plt.close()

# Append to table1 CSV
os.makedirs(RESULTS_DIR, exist_ok=True)
out_csv = os.path.join(RESULTS_DIR, "table1_inter_event.csv")
if not os.path.exists(out_csv):
    with open(out_csv, "w") as f:
        f.write("anion,T,beta,beta_se,prefactor,prefactor_se,mean_tau_ps,median_tau_ps,p95_tau_ps\n")
with open(out_csv, "a") as f:
    f.write(f"{anion},{T},{coeff},{coeff_se},{inter},{inter_se},"
            f"{np.average(sur):.3f},{np.median(sur):.1f},{np.quantile(sur,0.95):.1f}\n")
print(f"beta={coeff:.4f} +/- {coeff_se:.4f},  mean_tau={np.average(sur):.2f} ps")
