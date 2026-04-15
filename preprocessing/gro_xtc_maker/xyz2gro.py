"""
Convert center-of-mass XYZ trajectory files to GROMACS GRO format, mapping
atom names and box dimensions for use with MDAnalysis and GROMACS tools.
"""
import numpy as np
import os, sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model, T = input("").split() ; T = int(T)
system_dir = os.path.join(DATA_ROOT, anion_model, str(T))
trj_path = os.path.join(system_dir, "center.xyz")
out_path = os.path.join(system_dir, "center.gro")
Natoms = 200
Nsteps = int(os.popen(f"tail -n {Natoms+1} {trj_path} | head -n 1").readline().split()[2])
head = 2
trj = open(trj_path, 'r')
gro = open(out_path, 'w')
for t in range(Nsteps):
    step = t+1
    if step%1000 == 0:
        print(f"Converting XYZ to GRO {step:5d}/{Nsteps:5d} steps...")
    gro.write(f"COM Traj t= {t*1.5:.4f} step= {t}\n")
    gro.write(f"  {Natoms}\n")
    for i in range(head):
        line = trj.readline().split()
        if i == 1:
            boxl = float(line[4])*0.1
    resnr = 1
    for i in range(Natoms):
        line = trj.readline().split()
        atnm = line[0]
        atnr = resnr
        if atnm == "Li": resnm = "LI" ; name = "Li"
        elif atnm == "TF": resnm = "TFS" ; name = "F"
        elif atnm == "PY": resnm = "PYR" ; name = "P"
        x = float(line[1])*0.1
        y = float(line[2])*0.1
        z = float(line[3])*0.1
        gro.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(resnr, resnm, name, atnr, x, y, z))
        resnr += 1
    gro.write(f"   {boxl:.5f}   {boxl:.5f}   {boxl:.5f}\n")
gro.close()
trj.close()
xtc_path = os.path.join(system_dir, "center.xtc")
os.system(f"echo 0 | gmx_mpi trjconv -f {out_path} -s {out_path} -o {xtc_path}")
