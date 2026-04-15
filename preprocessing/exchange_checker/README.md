# exchange_checker

Detects solvation shell exchange events from center-of-mass trajectories.

For each timestep and each Li⁺ atom, records which anion molecules are
within the cutoff radius `r_cut` — producing a binary contact matrix
`shell_exchange.txt`.

## Files

| File | Purpose |
|------|---------|
| `shell_exchange.py` | Main script: reads center.xyz, outputs shell_exchange.txt |
| `plot.py` | Visualize exchange event statistics |

## Usage

```bash
python shell_exchange.py <anion> <T>
# e.g. python shell_exchange.py fsi 298
```

Input: `{DATA_ROOT}/{anion}/{T}/center.xyz`  
Output: `{DATA_ROOT}/{anion}/{T}/shell_exchange.txt`

Format of `shell_exchange.txt`: binary matrix, shape = (Nsteps × N_atoms).
Row t, column i = 1 if anion i is in the Li⁺ solvation shell at time t.
