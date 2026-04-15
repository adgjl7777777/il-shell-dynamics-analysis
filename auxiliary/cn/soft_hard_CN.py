"""
Compute and plot the coordination number distribution broken down by soft and
hard solvation states for each anion model and temperature.
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib
from tqdm import trange
anion= sys.argv[1]
trail=1.0
#max_ylist= {"298":70,"353":40,"373":30,"423":20}
import time

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import CLASSIFY_DIR, DATA_ROOT

CN=np.zeros((2,10,4))
color= ["#5555FF","#55FF55","#FFAA55","#FF5555"]

for k , T in enumerate(["298","353","373","423"]):
    out_path = os.path.join(DATA_ROOT, anion, str(T), "pair_check")
    for i in range(5):
        with open(os.path.join(CLASSIFY_DIR, "result", anion, "hard", str(T), f"{trail}_{i}.txt"),"r") as f:
            hard = [[int(j) for j in i.strip().split(" ") if j.isdigit()] for i in f.readlines()] 
        soft = []
        with open(os.path.join(CLASSIFY_DIR, "result", anion, "soft", str(T), f"{trail}_{i}.txt"),"r") as f:
            soft = [[int(j) for j in i.strip().split(" ") if j.isdigit()] for i in f.readlines()]
        hard = [[j[0],j[-1], 1] for j in hard]
        soft = [[j[0],j[-1], 0] for j in soft]
        tots = sorted(hard+soft)
        tots.append([10000000,10000000,2])
        tots[-1][-1] = tots[-2][-1]
        totsindex = 0
        with open(os.path.join(out_path, f"{i}.txt"),"r") as f:
            l = f.readlines()
            for j in l:
                t = int(j.split(",")[0])
                if t>100000:
                    break
                if tots[totsindex][1] <= t:
                    totsindex += 1
                CN[tots[totsindex][2]][len(j.split(","))-2][k]+=1


CCN = CN[0]

BN = np.divide(CCN,np.sum(CCN,axis=0))
for k, T in enumerate(["298","353","373","423"]):
    base=  np.array([2,3,4,5])
    plt.bar(base-(3-2*k)*0.1,BN[2:6,k], color = color[k],label=f"{T}K",width=0.2)

plt.ylabel("Ratio")
plt.xlabel("Coordnation Number")
plt.xlim([1.5,6.5])
plt.xticks([2,3,4,5,6])
plt.ylim([0,1])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.legend()

plt.savefig(f"{anion}_soft.png",dpi=300)

plt.cla()
plt.clf()
plt.close()


CCN = CN[1]

BN = np.divide(CCN,np.sum(CCN,axis=0))
for k, T in enumerate(["298","353","373","423"]):
    base=  np.array([2,3,4,5])
    plt.bar(base-(3-2*k)*0.1,BN[2:6,k], color = color[k],label=f"{T}K",width=0.2)

plt.ylabel("Ratio")
plt.xlabel("Coordnation Number")
plt.xlim([1.5,6.5])
plt.xticks([2,3,4,5,6])
plt.ylim([0,1])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.legend()

plt.savefig(f"{anion}_hard.png",dpi=300)


plt.cla()
plt.clf()
plt.close()
