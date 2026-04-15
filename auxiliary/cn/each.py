"""
Calculate the per-Li-atom coordination number by scanning pair distances in
the center-of-mass trajectory and writing per-atom shell membership to disk.
"""
import numpy as np
import os, sys
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model = sys.argv[1]
T = sys.argv[2]
trj_path = os.path.join(DATA_ROOT, anion_model, str(T), "center.xyz")
out_path = os.path.join(DATA_ROOT, anion_model, str(T), "pair_check")
os.makedirs(out_path,exist_ok=True)
Nskips = 2
dim = 3
Natoms = 200
Nsteps = int(os.popen(f"tail -n {Natoms+1} {trj_path} | head -n 1").readline().split()[2])
Nli = 5 ; Nanions = 100 ; Ncations = 95
basecut = {"fsi":5.6,"tfsi":5.55,"beti":5.8}
rcut = basecut[anion_model]
 
cation_name = "PY"
anion_name = "TF"

boxl = np.zeros(Nsteps, float)
li_crd = np.zeros((Nsteps, Nli, dim), float)
anion_crd = np.zeros((Nsteps, Nanions, dim), float)
cation_crd = np.zeros((Nsteps, Ncations, dim), float)

trj = open(trj_path,'r')
# Reading Trajectories..
for t in trange(Nsteps):
    step = t+1
#    if step%100 == 0:
#        print(f"Reading trajectories... {step}/{Nsteps} steps...")
    for i in range(Nskips):
        line = trj.readline().split()
        if i == 0:
            Natoms = 200
        elif i == 1:
            boxl[t] = float(line[4])
    Nli = 0 ; Nan = 0 ; Nca = 0
    for i in range(Natoms):
        line = trj.readline().split()
        atnm = line[0]
        x = float(line[1])
        y = float(line[2])
        z = float(line[3])
        if atnm == "Li":
            li_crd[t, Nli, 0] = x
            li_crd[t, Nli, 1] = y
            li_crd[t, Nli, 2] = z
            Nli += 1
        elif atnm == "TF":
            anion_crd[t, Nan, 0] = x
            anion_crd[t, Nan, 1] = y
            anion_crd[t, Nan, 2] = z
            Nan += 1
        elif atnm == "PY":
            cation_crd[t, Nca, 0] = x
            cation_crd[t, Nca, 1] = y
            cation_crd[t, Nca, 2] = z
            Nca += 1
trj.close()

# Calculating Nearest Neighbors...
NN = []
for t in trange(Nsteps):
    step = t+1
#    if step%100 == 0:
#        print(f"Calculating nearest neighbors && shell exchanges... {step}/{Nsteps} steps...")
    NN.append([])
    for i in range(Nli):
        xli = li_crd[t,i,0]
        yli = li_crd[t,i,1]
        zli = li_crd[t,i,2]
        NN[t].append([])
        for j in range(Nan):
            xan = anion_crd[t,j,0]
            yan = anion_crd[t,j,1]
            zan = anion_crd[t,j,2]

            dx = xli - xan
            if abs(dx) > boxl[t]/2.0:
                dx -= boxl[t]*round(dx/boxl[t])
            dy = yli - yan
            if abs(dy) > boxl[t]/2.0:
                dy -= boxl[t]*round(dy/boxl[t])
            dz = zli - zan
            if abs(dz) > boxl[t]/2.0:
                dz -= boxl[t]*round(dz/boxl[t])

            dr = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dr < rcut:
                NN[t][i].append(j)

for i in range(Nli):
    f = open(out_path+f"{i}.txt","w")
    for t in trange(Nsteps):
        NN[t][i] = sorted(NN[t][i])
        f.write(f"{t},     ")
        for j in NN[t][i]:
            f.write(f"{j}, ")
        f.write("\n")
    f.close()
