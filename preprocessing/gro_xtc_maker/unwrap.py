"""
Unwrap PBC-wrapped center-of-mass trajectories by accumulating periodic
boundary crossings, then write the result to GROMACS GRO and XTC files for
downstream MDAnalysis-based analysis.
"""
import numpy as np
import os, sys
from tqdm import trange
import copy

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model, T = sys.argv[1:]
head = 2
Natoms = 200
adder = np.zeros((Natoms,3))
prev = np.ones((Natoms,3))
now = np.ones((Natoms,3))
Nsteps = 100001

trj_path = os.path.join(DATA_ROOT, anion_model, str(T), "center.xyz")
unwrap_dir = os.path.join(DATA_ROOT, "code", "gro_xtc_maker", "unwrap")
os.makedirs(unwrap_dir, exist_ok=True)
out_path = os.path.join(unwrap_dir, f"{anion_model}_{T}.gro")
xtc_path = os.path.join(unwrap_dir, f"{anion_model}_{T}.xtc")
box = np.array(os.popen(f"head -n 2 {trj_path} | tail -n 1 ").readline().split()[-3:],np.float64)
cut = 10

pyrtfsi = 322.39
litfsi = 287.075
libeti = 387.10
lifsi = 187.07
li = 6.941
tfsi = litfsi-li
pyr = pyrtfsi-tfsi
fsi = lifsi-li
beti = libeti-li


trj = open(trj_path, 'r')
gro = open(out_path, 'w')
for t in trange(Nsteps):
    gro.write(f"COM Traj t= {t:.4f} step= {t}\n")
    gro.write(f"  {Natoms}\n")
    for i in range(head):
        line = trj.readline().split()
    resnr = 1
    for i in range(Natoms):
        line = trj.readline().split()
        atnm = line[0]
        atnr = resnr
        if atnm == "Li": resnm = "LI" ; name = "Li"
        elif atnm == "TF": resnm = "ANI" ; name = "A"
        elif atnm == "PY": resnm = "PYR" ; name = "P"
        unwrapped = np.zeros(3)
        for coords in range(3):
            r = float(line[coords+1])
            if r< cut:
                now[i,coords] = 0
            elif r > box[coords]-cut:
                now[i,coords] = 2
            else:
                now[i,coords] = 1
            if prev[i,coords] == 0 and now[i,coords] == 2:
                adder[i,coords] -= 1
            elif prev[i,coords] == 2 and now[i,coords] == 0:
                adder[i,coords] += 1
            unwrapped[coords] = r + adder[i,coords]*box[coords]
        unwrapped /= 10
        gro.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(resnr, resnm, name, atnr, unwrapped[0], unwrapped[1], unwrapped[2]))
        resnr += 1
    gro.write(f"   {box[0]:.5f}   {box[1]:.5f}   {box[2]:.5f}\n")
    prev = copy.deepcopy(now)
gro.close()
trj.close()

os.system(f"echo 0 | gmx_mpi trjconv -f {out_path} -s {out_path} -o {xtc_path}")
