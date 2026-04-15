import numpy as np
import matplotlib.pyplot as plt
import os
import sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import DATA_ROOT

anion = "beti"
#T = [298,353,373,423]
color=["#000000","#333333","#666666","#999999","#BBBBBB"]
plt.figure(figsize=(15,8))
for j,T in enumerate([298, 353, 373, 423]):
	data = np.loadtxt(os.path.join(DATA_ROOT, anion, str(T), "shell_exchange.txt"), int)
	plt.subplot(4,1,j+1)
	for i in range(1,2):
		markerline, stemlines, baseline = plt.stem(data[:10000,0], data[:10000,i+1], basefmt="k-", markerfmt=" ", linefmt="k-")
		plt.setp(stemlines, 'color',color[i])
	plt.gca().axes.xaxis.set_visible(False)
	plt.yticks([0,1])
	plt.ylabel(f"{T}K")
plt.gca().axes.xaxis.set_visible(True)
	
plt.xlabel(f"Timeline (ps)")
plt.savefig(f"./exchange_{anion}.png",dpi=600)
