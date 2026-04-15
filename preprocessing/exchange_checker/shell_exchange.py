"""
Detect solvation shell exchange events for each Li ion by tracking which
anions enter or leave the cutoff radius at each trajectory frame, writing
the exchange event list to shell_exchange.txt.
"""
import numpy as np
import os, sys
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model, T = input("").split()
trj_path = os.path.join(DATA_ROOT, anion_model, str(T), "center.xyz")
out_path = os.path.join(DATA_ROOT, anion_model, str(T), "shell_exchange.txt")
Natoms = 200
Nsteps = int(os.popen(f"tail -n {Natoms+1} {trj_path} | head -n 1").readline().split()[2])
Nskips = 2
dim = 3
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
# Reading Trajectories...
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
#print(step)
#5.8
#5.55
#5.6
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

# Calculating shell exchanges
# If the member of anions around a litihum at time t is different from time t',
# we say there is a shell exchange.
shell_exchanges = np.zeros((Nsteps, Nli), int)
for t in trange(Nsteps):
    step = t+1
    tt = max(t-1,0) # previous time
    for i in range(Nli):
        Nt = sorted(NN[t][i])
        Ntt = sorted(NN[tt][i])
        if Nt != Ntt:
            shell_exchanges[t,i] = 1

# Writing outputs
out = open(out_path, 'w')
for t in trange(Nsteps):
    step = t+1
#    if step%1000 == 0:
#        print(f"Writing outputs {step}/{Nsteps} steps...")
    out.write(f"{t}    ")
    for i in range(Nli):
        out.write(f"{shell_exchanges[t,i]} ")
    out.write('\n')
out.close()
