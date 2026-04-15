"""
Center-of-mass preprocessing pipeline for FSI/BETI anion systems, reading
raw trajectory files and computing per-molecule COM coordinates saved to
center.xyz.
"""
import numpy as np
import sys
import os
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, RAW_DATA_ROOT

T = sys.argv[1]
anion = sys.argv[2]
aaaa = anion.upper()
#basedir = os.path.join(RAW_DATA_ROOT, "wmi-md", "NVT", aaaa, str(T))
#os.chdir(basedir)
targetdir = os.path.join(DATA_ROOT, anion, str(T))
os.makedirs(targetdir,exist_ok=True)
f = open(os.path.join(targetdir, "center.xyz"),"w")
prefix = targetdir # file name

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
tfsi_mass = {"bf4":np.array([8.298, 18.998, 18.998, 18.998, 18.998]),
             "fsi":np.array([14.007, 32.066, 15.999, 15.999, 18.998, 32.066, 15.999, 15.999, 18.998, 0.000]),
             "pf6":np.array([30.974, 18.998, 18.998, 18.998, 18.998, 18.998, 18.998]),
             "tfsi":np.array([14.007, 32.066, 32.066, 15.999, 15.999, 15.999, 15.999, 12.011, 12.011, 18.998, 18.998, 18.998, 18.998, 18.998, 18.998, 0.000]),
             "beti":np.array([15.999, 32.066, 15.999, 14.007, 32.066, 15.999, 15.999, 12.011, 18.998, 18.998, 12.011, 18.998, 18.998, 18.998, 12.011, 18.998, 18.998, 12.011, 18.998, 18.998, 18.998, 0.000])}
pyr_mass = np.array([14.007, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 1.008, 1.008, 1.008,\
             1.008,  1.008,  1.008,  1.008,  1.008,  1.008,  1.008, 1.008, 1.008, 1.008,\
            12.011,  1.008,  1.008, 12.011,  1.008,  1.008, 12.011, 1.008, 1.008, 1.008])

connectivity = {"bf4": "BFFFF",
                "fsi": "NS-O=O=FsS-O=O=FsLp",
                "pf6": "P-FpFpFpFpFpFp",
                "tfsi":"NS-S-O=O=O=O=CmCmFFFFFFLp",
                "beti":"O=S-O=NS-O=O=CFFCmFFFCFFCmFFFLp"}
pyr_connectivity = "N+CCCCCmCHHHHHHHHHHHHHCHHCHHCmHHH"

# Essential for msd calculation
#w.SetMolTypes(['TFS', 'PYR', 'Li'], [16, 30, 1], [0, 0, 6.941])	# PEO
tmp_dir = os.path.join(DATA_ROOT, "code", "center_maker", "tmp", anion)
traj = np.load(os.path.join(tmp_dir, f"{T}_pos.npy"))
box = np.load(os.path.join(tmp_dir, f"{T}_box.npy"))
an = np.load(os.path.join(tmp_dir, f"{T}_an.npy"))
# Inclusion of H & Lp or not
steps = traj.shape[0]
atoms = traj.shape[1]
aM = sum(tfsi_mass[anion])
cM = sum(pyr_mass)
al = len(tfsi_mass[anion])
cl = len(pyr_mass)
for step in trange(steps):
    pyr_com=np.zeros((95,3))
    li_r=traj[step,0:5]
    r_com=np.zeros((100,3))
    pluser=5
    for astep in range(100):
        at = traj[step,pluser+astep*al:pluser+(astep+1)*al]
        for ijk in range(3):
            if max(at[:,ijk])>box[ijk]-5 and min(at[:,ijk])<5:
                at[:,ijk]=np.where(at[:,ijk]> box[ijk]/2,at[:,ijk],at[:,ijk]+box[ijk])
        r_com[astep]=np.sum(at*np.tile(tfsi_mass[anion],reps=[3,1]).T/aM,axis=0)%box
    pluser=5+100*al
    for cstep in range(95):
        ct = traj[step,pluser+cstep*cl:pluser+(cstep+1)*cl]
        for ijk in range(3):
            if max(ct[:,ijk])>box[ijk]-5 and min(ct[:,ijk])<5:
                ct[:,ijk]=np.where(ct[:,ijk]> box[ijk]/2,ct[:,ijk],ct[:,ijk]+box[ijk])
        pyr_com[cstep]=np.sum(ct*np.tile(pyr_mass,reps=[3,1]).T/cM,axis=0)%box
    f.write(f'{atoms}\n')
    f.write(f'Step = {step} Box: {box[0]:8.3f} {box[1]:8.3f} {box[2]:8.3f}\n')
    for j in li_r:
        f.write(f'Li\t{j[0]:8.8f}\t{j[1]:8.8f}\t{j[2]:8.8f}\n')
    for j in r_com:
        f.write(f'TF\t{j[0]:8.8f}\t{j[1]:8.8f}\t{j[2]:8.8f}\n')
    for j in pyr_com:
        f.write(f'PY\t{j[0]:8.8f}\t{j[1]:8.8f}\t{j[2]:8.8f}\n')
    if step % 1000 == 0:
        f.close()
        f = open(os.path.join(targetdir, "center.xyz"),"a")

f.close()
