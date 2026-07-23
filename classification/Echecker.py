"""
Two-state soft/hard classification of Li+ solvation dynamics.

Reads shell_exchange.txt (binary matrix: rows=timesteps, cols=Li atoms)
and classifies each Li-anion pair trajectory into soft or hard state
using the mean inter-event time Δt from TABLE1 as the threshold.

Algorithm (see paper §Classification of Kinetic States):
  Step 1: Identify shell change events
  Step 2: Use Δt = TABLE1[anion][T].delta_t_exp as the correlation interval
  Step 3: Events within Δt → soft state
  Step 4: Gaps between soft segments < 2Δt → merge into one soft segment
  Step 5: Remaining intervals → hard state

Output: result/{anion}/{soft|hard}/{T}/1.0_{i}.txt

Usage:
    python Echecker.py <anion>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, TABLE1

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

anion     = sys.argv[1]
threshold = 1        # canonical event-density threshold
multi     = 1.0      # canonical trail (merge criterion multiplier)

plt.rc('font', size=20)

# Δt from Table 1: mean inter-event time, used as classification threshold
inter = {str(T): TABLE1[anion][T][2] for T in [298, 353, 373, 423]}

max_ylist = {"298": 70, "353": 40, "373": 30, "423": 20}
multi2 = 0
multi3 = 2*multi

for T in ["298","353","373","423"]:
    se_path = f"{DATA_ROOT}/{anion}/{T}/shell_exchange.txt"
    se = np.loadtxt(se_path, int)
    Nsteps, Natoms = np.shape(se)
    axs = ["" for _ in range(Natoms-1)]
    os.makedirs(f"Factor/{anion}/{T}",exist_ok=True)
    os.makedirs(f"Data/{anion}/{T}",exist_ok=True)
    os.makedirs(f"result/{anion}/soft/{T}",exist_ok=True)
    os.makedirs(f"result/{anion}/hard/{T}",exist_ok=True)

    plt.figure(figsize=(24,24))
    for cur in range(Natoms-1):
        
        Flag2 = True
        
        train = False
        tlist = []
        typelist = []
        trainlist = []
        viewlistsoft = []
        viewlisthard = []
        truetrainlist=[]
        if cur != 0:
            axs[cur]=plt.subplot(5,1,cur+1,sharex=axs[0],sharey=axs[0])
        else:
            axs[cur]=plt.subplot(5,1,cur+1)
        checker=0
        for t in range(Nsteps):
            if t == 0:
                print(f"Calculating E flow: {T}K cluster {cur+1}")
            if se[t,cur+1]==1 and not train: #train 아닐 때 이벤트 발생하면 새로 트레인 만들기
                trainlist.append([t])
                train=True
                tlist.append([t,0])
                checker=t
                Flag2=False
            if len(tlist) > 0 and t-checker>=inter[T]*multi and train:
                if len(trainlist[-1]) == 0: #딱 한번만 이벤트 발생할 때 고려
                    trainlist[-1].append(checker)
                if len(trainlist[-1])/(t-tlist[-1][0]-inter[T]*multi)*inter[T]>threshold:
                    #+inter[T]*multi
                    typelist.append(True) #Soft
                    truetrainlist.append(trainlist[-1])
                else:
                    typelist.append(False) #hard
                    truetrainlist.append(trainlist[-1])
                train=False
                tlist[-1][1]=trainlist[-1][-1]
                tt = tlist[-1][0]
                views = 0
                for eachs in trainlist[-1]:
                    while(True):
                        if tt +inter[T] > eachs:
                            views +=1
                            break
                        else:
                            if typelist[-1]:
                                viewlistsoft.append([tt,views, 0])
                            else:
                                viewlisthard.append([tt,views, 1])
                            views=0
                            tt += inter[T]
                if views != 0:
                    if typelist[-1]:
                        viewlistsoft.append([tt,views, 0])
                    else:
                        viewlisthard.append([tt,views, 1])
                if t > 100000: #100000번 까지만 확인
                    break
                if se[t,cur+1]==1 and not train: #train 아닐 때 이벤트 발생하면 새로 트레인 만들기
                    trainlist.append([t])
                    train=True
                    tlist.append([t,0])
                    checker=t
                    Flag2=False
            if se[t,cur+1]==1 and train and Flag2: #트레인 맞을 때 추가 
                trainlist[-1].append(t)
                checker=t
            Flag2=True
            if len(tlist) > 0 and tlist[-1][-1] != 0 and t > 100000: # 100000번 까지만 확인
                break
        t1_train = []
        t1_type = []
        print("2",end=" ", flush=True) #이벤트 사이사이 결정
        for i in trange(len(truetrainlist)-1):
            t1_train.append(truetrainlist[i])
            t1_type.append(typelist[i])
            t1_train.append([truetrainlist[i][-1],truetrainlist[i+1][0]])
            t1_type.append(True if typelist[i] and typelist[i+1] and truetrainlist[i+1][0]-truetrainlist[i][-1] <=multi3*inter[T] else False)
        t1_train.append(truetrainlist[-1])
        t1_type.append(typelist[-1])
        if t1_train[0][0] != 0:
            if t1_type[0] and t1_train[0][0] < multi*inter[T]:
                t1_train.insert(0,[0,t1_train[0][0]])
                t1_type.insert(0,True)
            else:
                t1_train.insert(0,[0,t1_train[0][0]])
                t1_type.insert(0,False)
        if t1_train[-1][-1] != 100000:
            if t1_type[-1] and 100000-t1_train[-1][-1] < multi*inter[T]:
                t1_train.append([t1_train[-1][-1],100000])
                t1_type.append(True)
            else:
                t1_train.append([t1_train[-1][-1],100000])
                t1_type.append(False)
        i = 0
        same = []
        print(t1_train, flush=True)
        print(t1_type, flush=True) 
        t2_train=[]
        t2_type=[]
        print("3",end=" ", flush=True) # 앞과 뒤 type이 같으면 같은 것으로 가정
        for i in trange(len(t1_train)):
            if len(same) == 0:
                same.append(i)
            if i+1 < len(t1_train) and t1_type[i] == t1_type[i+1]:
                same.append(i+1)
            else:
                t2_train.append([])
                for j in same:
                    t2_train[-1].extend(t1_train[j])
                t2_type.append(t1_type[i])
                same = []
        if len(same) != 0:
            t2_train.append([])
            for j in same:
                t2_train[-1].extend(t1_train[j])
            t2_type.append(t1_type[i])
        print(t2_train, flush=True)
        print(t2_type, flush=True)
                
                
        t3_train=[]
        t3_type=[]
        print("4",end=" ", flush=True)
        for i in trange(len(t2_train)):
            if t2_train[i][-1]-t2_train[i][0] != 0:
                t3_train.append(t2_train[i])
                t3_type.append(t2_type[i])
        print(t3_train, flush=True)
        print(t3_type, flush=True)
                
        t4_train=[]
        t4_type=[]
        print("5",end=" ", flush=True)
        for i in trange(len(t3_train)):
            if len(same) == 0:
                same.append(i)
            if i+1 < len(t3_train) and t3_type[i] == t3_type[i+1]:
                same.append(i+1)
            else:
                t4_train.append([])
                for j in same:
                    t4_train[-1].extend(t3_train[j])
                t4_type.append(t3_type[i])
                same = []
        if len(same) != 0:
            a=3
        print(t4_train, flush=True)
        print(t4_type, flush=True)
                
                
        soft = []
        hard = []
        spacelist = [[],[]]
        print("6",end=" ", flush=True)
        for i in trange(len(t4_train)): 
            if t4_type[i]:
                soft.append(sorted(list(set(t4_train[i]))))
            else:
                hard.append(sorted(list(set(t4_train[i]))))
        print("7", flush=True)
        with open(f"result/{anion}/soft/{T}/{multi}_{cur}.txt", "w") as f:
            for j in soft:
                spacelist[0].append([j[0],j[-1]])
                for k in j:
                    f.write(str(int(k)))
                    f.write(" ")
                f.write("\n")
        with open(f"result/{anion}/hard/{T}/{multi}_{cur}.txt", "w") as f:
            for j in hard:
                spacelist[1].append([j[0],j[-1]])
                for k in j:
                    f.write(str(int(k)))
                    f.write(" ")
                f.write("\n")
            
        viewlistsoft = np.array(viewlistsoft)
        viewlisthard = np.array(viewlisthard)
        if len(viewlistsoft)>0:
            plt.stem(viewlistsoft[:,0],viewlistsoft[:,1], basefmt=" ", markerfmt=" ", linefmt="b-")
        if len(viewlisthard)>0:
            plt.stem(viewlisthard[:,0],viewlisthard[:,1], basefmt=" ", markerfmt=" ", linefmt="r-")
        #plt.hlines(0,0,100000,colors="black",linestyle="solid")

        max_y=max_ylist[T]
        #max_y = max(max(viewlistsoft[:,1]),max(viewlisthard[:,1]))
    
        softr = 0
        hardr = 0
        for spacer in spacelist[0]:
            softr += spacer[1]-spacer[0]
            plt.fill([spacer[0],spacer[1],spacer[1],spacer[0]],[0,0,max_y,max_y],color="blue",alpha=0.1)
        for spacer in spacelist[1]:
            hardr += spacer[1]-spacer[0]
            plt.fill([spacer[0],spacer[1],spacer[1],spacer[0]],[0,0,max_y,max_y],color="red",alpha=0.1)
        with open(f"Data/{anion}/{T}/{multi}.txt","a") as f:
            f.write(f"{cur}, {softr}, {hardr}, {softr/(softr+hardr)}\n")
        if cur == 0:
            plt.title(f"Timeline of {T}K")
            plt.xlim([0,100000])
            plt.ylim([0,max_y])
            plt.xticks([_*10000 for _ in range(11)])
        elif cur == 2:
            plt.ylabel(r"Events ($E_{\mathrm{" + anion.upper() + r"}}$)")
        elif cur == 4:
            plt.xlabel("Timeline(ps)")
    plt.tight_layout()
    plt.savefig(f"Factor/{anion}/{T}/{threshold}_{multi}_{anion}_{T}K.png",dpi=300)
    #plt.savefig(os.path.join(DATA_ROOT, anion, str(T), f"Eflow_{i}.png"),dpi=600)
    plt.clf()
