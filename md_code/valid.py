import numpy as np
import matplotlib.pyplot as plt
path = "/nas_2/transcendence/cowork/wmi-md/NPT/Temp"

T = ["298","353","373","423"]


color= ["#5555FF","#55FF55","#FFAA55","#FF5555"]
for it in range(len(T)):
    z=[]
    with open(path+"/"+T[it]+"/fort.65","r") as f:
        for _ in range(6):
            __ = f.readline()
        z = f.readlines()
    temp = []
    size = []
    for j in z:
        dump = j.strip().split()
        temp.append(float(dump[1]))
        size.append(float(dump[3]))
    plt.scatter(temp,size,color=color[it])
    plt.xlim([int(T[it])-2, int(T[it])+2])
    plt.xticks(np.arange(int(T[it])-2, int(T[it])+2, 1.0))
    #plt.yticks(np.arange(min(size), max(size)+1, 1.0))
    plt.savefig(f"{T[it]}.png",dpi=600)
    plt.clf()