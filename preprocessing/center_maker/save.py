"""
Save raw trajectory coordinates and box dimensions for FSI/BETI systems to
numpy arrays via wmipost, serving as the initial data-extraction step in the
preprocessing pipeline.
"""
import wmipost as wp
import numpy as np
import sys
import os
from tqdm import trange

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT, RAW_DATA_ROOT

T = os.environ["T"]
anion = os.environ["anion"]

basedir = os.path.join(RAW_DATA_ROOT, "wmi-md", "NVT", anion.upper(), str(T))
os.chdir(basedir)
targetdir = os.path.join(DATA_ROOT, anion, str(T))
os.makedirs(targetdir,exist_ok=True)
tmp_dir = os.path.join(DATA_ROOT, "code", "center_maker", "tmp", anion)
os.makedirs(tmp_dir,exist_ok=True)
f = open(os.path.join(targetdir, "center.xyz"),"w")
prefix = targetdir # file name

# An example file to use wmipost properly
w = wp.wmi()			# Opening a class
w.top()					# Reading files about topology : connectivity.dat, ff.dat, coords.out
w.trj('fort.77')		# Reading trajectory files : default name = 'fort.77'

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

np.save(os.path.join(tmp_dir, f"{T}_pos"),w.Atom.Pos)
np.save(os.path.join(tmp_dir, f"{T}_box"),w.Trj.Box[0])
np.save(os.path.join(tmp_dir, f"{T}_an"),w.Atom.RDtypes)
