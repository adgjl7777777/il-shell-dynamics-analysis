"""
Produce the real (physics-correct) unwrapped trajectory by detecting and
correcting per-frame periodic boundary jumps in the center-of-mass XYZ file,
outputting GROMACS GRO and XTC files.
"""
import numpy as np
import os, sys
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model, T = sys.argv[1:]
head = 2
Natoms = 200
Nsteps = 100001

system_dir = os.path.join(DATA_ROOT, anion_model, str(T))
trj_path = os.path.join(system_dir, "center.xyz")
out_path = os.path.join(system_dir, "center_dum.gro")
un_out_path = os.path.join(system_dir, "center_unwrap.gro")
un_xtc_path = os.path.join(system_dir, "center_unwrap.xtc")
if os.path.isfile(un_out_path):
    os.remove(un_out_path)
if os.path.isfile(un_xtc_path):
    os.remove(un_xtc_path)

box = np.array(os.popen(f"head -n 2 {trj_path} | tail -n 1 ").readline().split()[-3:],np.float64)
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
        x = float(line[1])*0.1
        y = float(line[2])*0.1
        z = float(line[3])*0.1
        gro.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(resnr, resnm, name, atnr, x, y, z))
        resnr += 1
    gro.write(f"   {box[0]:.5f}   {box[1]:.5f}   {box[2]:.5f}\n")
gro.close()
trj.close()
os.system(f"echo 0 | gmx_mpi trjconv -f {out_path} -s {out_path} -pbc nojump -o {un_out_path}")
os.system(f"echo 0 | gmx_mpi trjconv -f {out_path} -s {out_path} -pbc nojump -o {un_xtc_path}")
os.remove(out_path)
