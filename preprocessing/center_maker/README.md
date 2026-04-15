# center_maker

Extracts center-of-mass (COM) coordinates for each ion from raw WMI-MD binary trajectory files.

## Pipeline

```
trajectory.out  (WMI-MD binary)
       │
save.py / save_tfsi.py   →  tmp/{anion}/{T}_pos.npy, _box.npy, _an.npy
       │
fsi_beti.py / tfsi.py    →  {anion}/{T}/center.xyz
```

## Files

| File | Purpose |
|------|---------|
| `save.py` | Load FSI/BETI binary trajectory → save pos/box/atomtype arrays as .npy |
| `save_tfsi.py` | Same as save.py for TFSI (different atom ordering) |
| `fsi_beti.py` | Compute COM from saved .npy arrays → write center.xyz (FSI/BETI) |
| `tfsi.py` | Same as fsi_beti.py for TFSI |
| `com.py` | Unified COM script (handles both, used in shell scripts) |
| `fort77.py` | Utility to parse fort.77 binary format |
| `atoms_steps_saver/saver.py` | Save atom count / step count metadata |

## Usage

```bash
# For FSI or BETI
export IL_RAW_DATA_ROOT=/path/to/raw/trajectories
export IL_DATA_ROOT=/path/to/preprocessed/md
export anion=fsi
export T=298
python save.py
python fsi_beti.py 298 fsi

# For TFSI
export IL_RAW_DATA_ROOT=/path/to/raw/trajectories
export IL_DATA_ROOT=/path/to/preprocessed/md
export T=298
python save_tfsi.py
python tfsi.py 298 tfsi
```

Typical input root: `$IL_RAW_DATA_ROOT/{source-layout}/...`

Output: `$IL_DATA_ROOT/{anion}/{T}/center.xyz`
