"""
Compute and plot the coordination number (CN) distribution of Li atoms for
all anion models and temperatures, producing the CN figure for the paper.
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model = sys.argv[1]
Temp = [298,353,373,423]
CN=np.zeros((10,4))
color= ["#5555FF","#55FF55","#FFAA55","#FF5555"]

for k, T in enumerate(Temp):
    out_path = os.path.join(DATA_ROOT, anion_model, str(T), "pair_check")
    Nli = 5
    for i in range(Nli):
        f = open(os.path.join(out_path, f"{i}.txt"),"r")
        l = f.readlines()
        for j in l:
            CN[len(j.split(","))-2][k]+=1
        f.close()
print(CN)
BN = np.divide(CN,np.sum(CN,axis=0))
for k, T in enumerate(Temp):
    base=  np.array([2,3,4,5])
    plt.bar(base-(3-2*k)*0.1,BN[2:6,k], color = color[k],label=f"{T}K",width=0.2)

plt.ylabel("Ratio")
plt.xlabel("Coordnation Number")
plt.xlim([1.5,6.5])
plt.xticks([2,3,4,5,6])
plt.ylim([0,1])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.legend()

plt.savefig(f"{anion_model}.png",dpi=300)
