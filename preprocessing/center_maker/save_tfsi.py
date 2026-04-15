"""
Save raw TFSI trajectory coordinates and box dimensions to numpy arrays,
reading directly from the LAMMPS-style trajectory file as the initial
data-extraction step for the TFSI preprocessing pipeline.
"""
import numpy as np
import sys
import os
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, RAW_DATA_ROOT

T = os.environ["T"]
#T = "298"
basedir = os.path.join(RAW_DATA_ROOT, "tfsi", "prd", f"{T}K")

Path = os.path.join(basedir, "trajectory.out")
trj = open(Path,'r')
Natoms = int(os.popen(f"head -n 1 {Path}").readline().split()[0])
Nsteps = int(os.popen(f"tail -n {Natoms+1} {Path} | head -n 1").readline().split()[2])

Box = np.zeros((Nsteps,3))
head = 2
an = []
coords = np.zeros((Nsteps,Natoms,3))
targets = []
for i in trange(Nsteps):
    for j in range(head):
        line = trj.readline().split()
        if j == 1:
            Box[i,0] = float(line[4])
            Box[i,1] = float(line[5])
            Box[i,2] = float(line[6])
    for j in range(Natoms):
        line = trj.readline().split()
        if i == 0:
            an.append(line[0])
        coords[i,j,0] = float(line[1])
        coords[i,j,1] = float(line[2])
        coords[i,j,2] = float(line[3])
    if T != "423":
        Liset = coords[i,0:5,:]
        TFSIset_original = coords[i,5+30*95:,:]
        TFSIset = []
        for ii in range(int(len(TFSIset_original)/16)):
            for jj in [8,12,14,13,1,5,6,0,2,3,4,7,9,11,10,15]:
                TFSIset.append(TFSIset_original[ii*16+jj,:])
        PYRset = coords[i,5:5+30*95,:]
        coords[i] = np.concatenate((Liset,TFSIset,PYRset),axis=0)
    else:
        Liset = coords[i,0:5,:]
        TFSIset = coords[i,5:5+16*100:,:]
        PYRset_original = coords[i,5+16*100:,:]
        PYRset = []
        for ii in range(int(len(PYRset_original)/30)):
            for jj in [4,5,8,6,7,9,3,19,20,26,25,22,21,23,24,28,29,27,17,18,2,16,15,1,14,13,0,11,10,12]:
                PYRset.append(PYRset_original[ii*30+jj,:])
        coords[i] = np.concatenate((Liset,TFSIset,PYRset),axis=0)
            
traj = coords

targetdir = os.path.join(DATA_ROOT, "tfsi", str(T))
os.makedirs(targetdir,exist_ok=True)
f = open(os.path.join(targetdir, "center.xyz"),"w")
prefix = targetdir
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
             #"tfsi":np.array([14.007, 32.066, 32.066, 15.999, 15.999, 15.999, 15.999, 12.011, 12.011, 18.998, 18.998, 18.998, 18.998, 18.998, 18.998, 0.000]),
             "tfsi":np.array([12.011, 18.998, 18.998, 18.998, 32.066, 15.999, 15.999, 14.007, 32.066, 15.999, 15.999, 12.011, 18.998, 18.998, 18.998, 0.000]),
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
anion="tfsi"

aM = sum(tfsi_mass[anion])
cM = sum(pyr_mass)
al = len(tfsi_mass[anion])
cl = len(pyr_mass)
box=Box[0]
for step in trange(Nsteps):
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
    f.write(f'{Natoms}\n')
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
