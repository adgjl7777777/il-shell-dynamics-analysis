import matplotlib.pyplot as plt
import re
import copy
import numpy as np

# --- Configuration ---
color = ["#5555FF", "#55FF55", "#FFAA55", "#FF5555"]
pattern = r"(\w+),(\d+),(\[\[.*\]\])"
# This list is for data lookup
myion = [1100, 2100, 2200, 3100, 3200, 3300, 4200, 4300, 4400, 5300, 5400, 5500]
# This list is for display on the x-axis
xtick_labels = ['11', '21', '22', '31', '32', '33', '42', '43', '44', '53', '54', '55']
Tdict = {298: 0, 353: 1, 373: 2, 423: 3}
ion_types = ["fsi", "tfsi", "beti"]
sh_types = ["soft", "hard", "total"]

# --- Data Loading ---
all_wowdict = {}
for sh in sh_types:
    basedict = {str(ions): np.zeros(4) for ions in myion}
    wowdict = {"fsi": copy.deepcopy(basedict), "tfsi": copy.deepcopy(basedict), "beti": copy.deepcopy(basedict)}
    try:
        with open(f"/nas_2/transcendence/_delete/cowork/my_work/code/coordinate_check/wow_{sh}.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(pattern, line.strip())
                if match:
                    anion = match.group(1)
                    T = int(match.group(2))
                    Datalist = eval(match.group(3))
                    data_dict = {item[0]: item[1] for item in Datalist}
                    if anion in wowdict:
                        for ion_code in basedict.keys():
                            wowdict[anion][ion_code][Tdict[T]] = data_dict.setdefault(ion_code, 0)
                else:
                    print(f"Could not parse line in wow_{sh}.txt: {line.strip()}")
    except FileNotFoundError:
        print(f"Warning: Data file wow_{sh}.txt not found. Plots for '{sh}' will be empty.")
    all_wowdict[sh] = wowdict

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(21, 15), sharex='col', sharey='row')

# Swapped loops: rows are now sh_types, columns are ion_types
for i, sh in enumerate(sh_types):
    for j, iontype in enumerate(ion_types):
        ax = axes[i, j]
        wowdict = all_wowdict[sh]
        
        for k, T in enumerate(Tdict.keys()):
            num = np.zeros(len(myion))
            for idx, ion_code in enumerate(myion):
                if iontype in wowdict and str(ion_code) in wowdict[iontype]:
                    num[idx] = wowdict[iontype][str(ion_code)][k]
            
            base = np.arange(1, len(myion) + 1)
            ax.bar(base - (3 - 2 * k) * 0.1, num, color=color[k], label=f"{T}K", width=0.2)
        
        #ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Add vertical lines
        ax.axvline(x=1.5, color='grey', linestyle='dotted')
        ax.axvline(x=3.5, color='grey', linestyle='dotted')
        ax.axvline(x=6.5, color='grey', linestyle='dotted')
        ax.axvline(x=9.5, color='grey', linestyle='dotted')

        ax.set_xlim([0.5, len(myion) + 0.5])
        if sh == "soft":
            ax.set_ylim([0,25000])
        else:
            ax.set_ylim([0,200000])

# --- Legend ---
# Place legend inside the top-right plot
axes[0, 0].legend(loc='upper left', fontsize=18)

# --- Axis Labels and Titles ---

# Column titles are now ion types
for j, iontype in enumerate(ion_types):
    axes[0, j].set_title(iontype.upper(), fontsize=30)

# Row labels (right side) are now sh types
for i, sh in enumerate(sh_types):
    ax_right = axes[i, 2]
    ax_right.yaxis.set_label_position("right")
    ax_right.set_ylabel(sh.capitalize(), rotation=270, labelpad=22, fontsize=24, va='center')

# Set x-tick labels only for the bottom row plots
plt.setp(axes, xticks=np.arange(1, len(myion) + 1))
for ax in axes[2, :]:
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=17.1)
for ax in axes[:, 0]:
    ax.tick_params(axis='y', labelsize=18.9) # Increased y-tick label size


# Shared, multi-line axis labels
# Y-axis: "Count" on the left
fig.text(0.07, 0.5, 'Count', va='center', rotation='vertical', fontsize=26.4)

# X-axis: "Coordination Info..." at the very bottom (updated text)
fig.text(0.5, 0.04, '[Coordination Number, Number of Anions in Coordinated Structure]', ha='center', fontsize=26.4)

# --- Final Adjustments and Save ---
plt.tight_layout(rect=[0.09, 0.06, 0.93, 0.94])
plt.savefig("/nas_2/transcendence/revision/exports/submission_package/main/Images/cndata.pdf", dpi=600)
plt.close(fig)

print("Successfully generated total.pdf with a 3x3 grid of plots.")
