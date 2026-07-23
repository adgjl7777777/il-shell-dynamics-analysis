import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib
anion = sys.argv[1]
plt.figure(figsize=(4.4,4.4))
T = [298,353,373,423]
interE  =[[],[],[],[]]
color= np.array([[5/16,5/16,1],[5/16,1,5/16],[1,5/8,5/16],[1,5/16,5/16]]) 
#marker= ['x','+','.',10]
marker= ['P','X','s','o']
for i in range(len(T)):
    for j in range(3):
        se_path = f"/nas_2/transcendence/_delete/cowork/my_work/code/total_real_plot/old/dE_vs_dt/data/{anion}/{T[i]}/Edt_{j}.txt"
        interE[i].append(np.loadtxt(se_path, float))
for i in range(len(T)):
    for j in [2,0,1]:
        x = interE[i][j][:,0]
        y = interE[i][j][:,1]
        visible = y > 1e-8
        if j == 1:  # Edt_1 = t: solid line
            plt.plot(x[visible], y[visible], "-", color=color[i], label=f"{T[i]} K")
        elif j == 0:  # Edt_0 = 2t: dotted line
            plt.plot(x[visible], y[visible], ":", color=color[i]*3/4, alpha=0.5)
        else:  # Edt_2 = t/2: dashed line
            plt.plot(x[visible], y[visible], "--", color=color[i]*3/4, alpha=0.5)
plt.yscale("log",base=10)
plt.xlim([1,500])
plt.ylim([10**(-5),1])
plt.xlabel("$E$")
plt.xscale("log",base=10)
plt.gca().set_xticks([1,5,10,50,100,500])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
plt.ylabel("$P_{\\Delta t}(E)$")
plt.title(anion.upper())

if anion == "fsi":
    plt.legend()
plt.tight_layout()
os.makedirs("/nas_2/transcendence/revision/exports/submission_package/main/Images/pevse", exist_ok=True)
plt.savefig(f"/nas_2/transcendence/revision/exports/submission_package/main/Images/pevse/{anion}_pow.pdf", dpi=600)
sys.exit(0)
for i in range(len(T)):
    for j in range(2,-1,-1):
        if j==0:
            plt.scatter(interE[i][j][:,0],interE[i][j][:,1],color=np.array([0,0,0])*j/4+color[i]*(4-j)/4,label=f"{T[i]} K",marker=marker[i],s=8)
        else:
            plt.scatter(interE[i][j][:,0],interE[i][j][:,1],color=np.array([0,0,0])*j/4+color[i]*(4-j)/4,marker=marker[i],s=8)
plt.yscale("log",base=10)
plt.xlim([0,120])
plt.ylim([10**(-5),1])
plt.xticks([0,20,40,60,80,100,120])
plt.xlabel("$E$")
plt.ylabel("$P_{\\Delta t}(E)$")

if anion == "fsi":
    plt.legend()
plt.savefig(f"./plot/{anion}_exp.png",dpi=600)
"""
print("Calculating the number of E in dt...")
EList = []
for i in range(Natoms):
    numN = int(Nsteps/dtList[i])
    init = 0
    for t in range(numN):
        if init == 0:
            end = dtList[i]
        else:
            end += dtList[i]
        E = sum(se[init:end,i])
        init = end
        EList.append(E)
pdf, x, _ = plt.hist(EList, bins = 30, density=True)
x = 0.5*(x[1:]+x[:-1])
out = open(f'pdf_{T}K.txt', 'w')
for i in range(len(x)):
    out.write(f"{x[i]:.8f}    {pdf[i]:.8f}\n")
out.close()

"""