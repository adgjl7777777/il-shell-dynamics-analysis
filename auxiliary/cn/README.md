# CN — Coordination Number Analysis

Computes the Li⁺ coordination number (average number of anions in the
first solvation shell) as a function of temperature and state (Fig CN).

## Files

| File | Purpose |
|------|---------|
| `each.py` | Per-Li-atom CN at each timestep |
| `CN.py` | Average CN over all Li atoms and time |
| `soft_hard_CN.py` | CN broken down by soft vs hard state |

## Usage

```bash
bash core.sh fsi 298
python CN.py fsi 298
python soft_hard_CN.py fsi 298
```
