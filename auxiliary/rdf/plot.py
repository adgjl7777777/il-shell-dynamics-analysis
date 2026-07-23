"""
Plot the radial distribution functions for all anion models across temperatures,
identifying the first minimum to define the coordination cutoff radius (Fig. 2).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

for anion in ["fsi","tfsi","beti"]:
	Temp = [298, 353, 373, 423]
	color= ["#5555FF","#55FF55","#FFAA55","#FF5555"]
	# Final-manuscript layout: enlarge the panel while retaining the original
	# type sizes, so the typography is proportionally more compact.
	plt.figure(figsize=(6.6, 6.16))
	for i,T in enumerate(Temp):
		data = np.loadtxt(os.path.join(DATA_ROOT, anion, str(T), "rdf.txt"), float)
		x = data[:,0] ; y = data[:,1]
		min=1000
		minx=0
		for j in range(len(x)):
			if x[j]>4 and x[j]<7 and min>y[j]:
				minx=x[j]
				min=y[j]
		plt.plot(x, y, label=f"{T} K", color=color[i], linewidth=3)
		with open("min.txt","a") as f:
			f.write(f"{anion}, {T}, {minx}, {min}\n")
	plt.xlabel(f"r ($\\AA$)", labelpad=5, fontsize=18)
	plt.ylabel(f"g(r)", labelpad=5, fontsize=18)
	plt.title(anion.upper(), fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	if anion == "fsi":
		plt.legend(fontsize=18, loc="best")
	plt.xlim(0,12)
	plt.tight_layout()
	plt.savefig(f"RDF2_Li{anion}.pdf", dpi=600)
	#plt.show()
	plt.cla()
	plt.clf()
	plt.close()
