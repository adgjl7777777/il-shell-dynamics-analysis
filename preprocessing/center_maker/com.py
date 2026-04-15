"""
Compute center-of-mass coordinates for FSI/BETI anions and cations from raw
MD trajectory data, writing the results to a center.xyz file.
"""
from numpy import *
from tqdm import trange
import os, sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, RAW_DATA_ROOT

anion_model, T = input("").split()
T = int(T)
#Path = f'../../{anion_model}/{T}K/skip.xyz'
#Path = f'../../../{anion_model}/prd/{T}K/trajectory.out'
Path = os.path.join(RAW_DATA_ROOT, anion_model, "prd", f"{T}K", "trajectory.out")
Natoms = int(os.popen(f"head -n 1 {Path}").readline().split()[0])
Nsteps = int(os.popen(f"tail -n {Natoms+1} {Path} | head -n 1").readline().split()[2])
#Nsteps = 1
head = 2
# Number of molecules
Ntfsi = 100 ; Npyr = 95 ; Nli = 5 


# Mass of the atoms in molecules
atom_mass = {"H": 1.008,
             "B": 8.298,
             "C": 12.011,
             "N": 14.007,
             "O": 15.999,
             "F": 18.998,
             "P": 30.974,
             "S": 32.066,
             "L": 0.000}
tfsi_mass = {"bf4":[8.298, 18.998, 18.998, 18.998, 18.998],
             "fsi":[14.007, 32.066, 15.999, 15.999, 18.998, 32.066, 15.999, 15.999, 18.998, 0.000],
             "pf6":[30.974, 18.998, 18.998, 18.998, 18.998, 18.998, 18.998],
             "tfsi":[14.007, 32.066, 32.066, 15.999, 15.999, 15.999, 15.999, 12.011, 12.011, 18.998, 18.998, 18.998, 18.998, 18.998, 18.998, 0.000]}
pyr_mass = [14.007, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 1.008, 1.008, 1.008,\
             1.008,  1.008,  1.008,  1.008,  1.008,  1.008,  1.008, 1.008, 1.008, 1.008,\
            12.011,  1.008,  1.008, 12.011,  1.008,  1.008, 12.011, 1.008, 1.008, 1.008]

connectivity = {"bf4": "BFFFF",
                "fsi": "NS-O=O=FsS-O=O=FsLp",
                "pf6": "P-FpFpFpFpFpFp",
                "tfsi":"NS-S-O=O=O=O=CmCmFFFFFFLp"}
pyr_connectivity = "N+CCCCCmCHHHHHHHHHHHHHCHHCHHCmHHH"

# Number of atoms in molecules
Ntfsi_atoms = len(tfsi_mass[anion_model]) ; Npyr_atoms = len(pyr_mass)

# Output file
output_dir = os.path.join(DATA_ROOT, anion_model, str(T))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "center.xyz")
file = open(output_path,'w')
# Reading trajectories and calculating the center of mass coordinates...
trj = open(f'{Path}','r')
for i in trange(Nsteps):
    tfsi_r = [] ; li_r = [] ; r_com = [] ; pyr_r = [] ; pyr_com = [] ; cg_r = []
    for j in range(head):
        line = trj.readline().split()
        if j == 1:
            xboxl = float(line[4])
            yboxl = float(line[5])
            zboxl = float(line[6])
    xi = 0 ; yi = 0 ; zi = 0
    tmp = ""
    for j in range(Natoms):
        line = trj.readline().split()
        atnm = line[0]
        x = float(line[1])
        y = float(line[2])
        z = float(line[3])
        if atnm == "Li":
            li_r.append([x, y, z])
        else:
            xi += x*atom_mass[atnm[:1]]
            yi += y*atom_mass[atnm[:1]]
            zi += z*atom_mass[atnm[:1]]
            tmp += atnm
            if sorted(tmp) == sorted(connectivity[anion_model]):
                M = sum(tfsi_mass[anion_model])
                x_com = xi/M
                y_com = yi/M
                z_com = zi/M
                r_com.append([x_com, y_com, z_com])
                xi = 0 ; yi = 0 ; zi = 0 ; tmp = ""
            elif sorted(tmp) == sorted(pyr_connectivity):
                M = sum(pyr_mass)
                x_com = xi/M
                y_com = yi/M
                z_com = zi/M
                pyr_com.append([x_com, y_com, z_com])
                xi = 0 ; yi = 0 ; zi = 0 ; tmp = ""
    file.write(f'{Nli + Ntfsi + Npyr}\n')
    file.write(f'Step = {Nsteps} Box: {xboxl:8.3f} {yboxl:8.3f} {zboxl:8.3f}\n')
    for j in range(Nli):
        file.write(f'Li\t{li_r[j][0]:8.8f}\t{li_r[j][1]:8.8f}\t{li_r[j][2]:8.8f}\n')
    for j in range(Ntfsi):
        file.write(f'TF\t{r_com[j][0]:8.8f}\t{r_com[j][1]:8.8f}\t{r_com[j][2]:8.8f}\n')
    for j in range(Npyr):
        file.write(f'PY\t{pyr_com[j][0]:8.8f}\t{pyr_com[j][1]:8.8f}\t{pyr_com[j][2]:8.8f}\n')
file.close()
trj.close()
