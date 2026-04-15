import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msds
import matplotlib.pyplot as plt
import sys
import os

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion_model = sys.argv[1]
T = sys.argv[2]
trail = 1.0
os.makedirs(f"plot/",exist_ok=True)
#for sh in ["soft","hard"]:
#    msd = np.loadtxt(os.path.join(DATA_ROOT, "code", "diffusion", "msd", anion_model, sh, str(T), f"{trail}.txt"))
#    time = np.arange(len(msd))
#    plt.loglog(time,msd)
#    plt.savefig(f"plot/{anion_model}/{T}/{sh}.png",dpi=300)
#    plt.clf()
#    plt.cla()
#    plt.close()

pyrtfsi = 322.39
litfsi = 287.075
libeti = 387.10
lifsi = 187.07
li = 6.941
tfsi = litfsi-li
pyr = pyrtfsi-tfsi
fsi = lifsi-li
beti = libeti-li
tpr = os.path.join(DATA_ROOT, "code", "gro_xtc_maker", "unwrap", f"{anion_model}_{T}.gro")
xyzz = os.path.join(DATA_ROOT, "code", "gro_xtc_maker", "unwrap", f"{anion_model}_{T}.xtc")
u = mda.Universe(tpr,xyzz, dt = 1.0)
print(u.atoms.types)

u.atoms[u.atoms.types == 'LI'].masses = li
u.atoms[u.atoms.types == 'A'].masses = beti
u.atoms[u.atoms.types == 'P'].masses = pyr
MSD = msds.EinsteinMSD(u, select='resname LI', msd_type='xyz', fft=True)
MSD.run()
##msdall = MSD.results.msds_by_particle
#msds =  np.mean(msdall,axis=1)
msds=MSD.results.timeseries
print(msds)
nframes = MSD.n_frames
timestep = 1 # this needs to be the actual time between frames
lagtimes = np.arange(nframes)*timestep # make the lag-time axis
plt.loglog(lagtimes, msds)
#plt.xlim([100,10000])
#plt.loglog(lagtimes, msd)
#plt.plot(np.arange(0,10001,100),msd[0:10001:100])
#print(msd[0:10000:100])
#plt.xlim([0,100])
plt.savefig(f"logplot/{anion_model}_{T}_total.png",dpi=300)
