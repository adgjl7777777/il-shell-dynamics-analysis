from numpy import *
from tqdm import trange
import os, sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, RAW_DATA_ROOT

anion_model, T = input("").split()
T = int(T)
#Path = f'../../{anion_model}/{T}K/skip.xyz'
#Path = f'../../../{anion_model}/prd/{T}K/trajectory.out'
Path = os.path.join(RAW_DATA_ROOT, anion_model, "prd", f"{T}K", "trajectory.out")
Natoms = int(os.popen(f"head -n 1 {Path}").readline().split()[0])
Nsteps = int(os.popen(f"tail -n {Natoms+1} {Path} | head -n 1").readline().split()[2])
output_dir = os.path.join(DATA_ROOT, anion_model, str(T))
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "info.txt"),'w') as f:
    f.write("Natoms\n")
    f.write(str(Natoms))
    f.write("\nNsteps\n")
    f.write(str(Nsteps))
    f.write("\n")
