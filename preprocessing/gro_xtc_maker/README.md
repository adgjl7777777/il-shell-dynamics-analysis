# gro_xtc_maker

Converts center.xyz to GROMACS GRO/XTC format and removes periodic boundary
condition (PBC) wrapping. The unwrapped trajectory is required for MSD analysis.

## Files

| File | Purpose |
|------|---------|
| `xyz2gro.py` | Convert center.xyz → center.gro + center.xtc |
| `unwrap.py` | Remove PBC wrapping from center.xtc → center_unwrap.xtc |
| `unwrap_real.py` | Final unwrap implementation (supersedes unwrap.py) |
| `diffusion.py` | Quick MSD test on the unwrapped trajectory |
| `plot.py` | Plot center-of-mass displacement traces |

## Usage

```bash
python xyz2gro.py <anion> <T>
python unwrap_real.py <anion> <T>
```

Output: `{DATA_ROOT}/{anion}/{T}/center.gro`, `center_unwrap.xtc`
