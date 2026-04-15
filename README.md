# Ionic-liquid shell-dynamics analysis code

This directory contains the analysis scripts used for Li+-anion
solvation-shell dynamics in Pyr14-based ionic-liquid molecular dynamics
trajectories. Starting from preprocessed MD outputs, the pipeline identifies
shell-change events, classifies each Li+ trajectory into soft and hard kinetic
states, fits state-resolved survival and residence-time distributions, computes
burstiness metrics, and writes CSV summaries under `results/`.

The scripts assume the directory layout used during the manuscript analysis.
Large trajectory-derived files are expected outside this directory. Set
`IL_DATA_ROOT` to point to those files before running the pipeline.

## Repository Layout

```text
code/
|-- config.py                    Shared paths, system lists, and fitted constants
|-- run_all.py                   Main pipeline runner
|-- run_arrhenius.py             Arrhenius-style analysis from fitted parameters
|-- run_ion_resolved.py          Per-Li+ soft fraction and residence-rate checks
|-- run_sensitivity.py           Threshold-multiplier sensitivity analysis
|-- run_bridging_timeline.py     Bridging-anion timeline figures
|
|-- preprocessing/               Scripts used to generate MD-derived inputs
|   |-- center_maker/            Li-centered coordinate preparation
|   |-- exchange_checker/        shell_exchange.txt generation
|   `-- gro_xtc_maker/           GRO/XTC and MSD preparation
|
|-- classification/              Soft/hard classification and pair splitting
|   |-- Echecker.py              Soft/hard state classifier
|   |-- soft_hard_split.py       Pair-trajectory splitting by state history
|   |-- f_pow.py                 Pair-survival power-law fits
|   |-- f_exp.py                 Auxiliary exponential fits
|   |-- dist.py                  Shared fitting utilities
|   |-- result/                  Generated soft/hard state intervals
|   |-- x/                       Generated pair categories
|   |-- pair/                    Pair-survival intermediate data
|   `-- event_collect/           Event data used for burstiness analysis
|
|-- analysis/
|   |-- table1/inter.py          Total shell-change inter-event statistics
|   |-- table2/theta.py          Soft-state fraction
|   |-- table2/h_fit.py          State residence-time exponential fits
|   |-- table2/f_pow.py          Pair-survival power-law fits
|   |-- table3/model_validate.py Synthetic-state validation calculation
|   |-- table4/                  Event collection and burstiness metrics
|   `-- table5/diffusion.py      State-resolved mobility estimates
|
|-- auxiliary/
|   |-- cn/                      Coordination-number analysis
|   `-- rdf/                     RDF analysis
|
`-- results/                     CSV outputs
```

Some directory and file names retain the internal table numbering used during
manuscript preparation.

## Requirements

Core analysis:

```bash
python -m pip install numpy scipy matplotlib tqdm
```

Optional scripts:

```bash
python -m pip install pandas MDAnalysis
```

`pandas` is used by the sensitivity and ion-resolved helper scripts.
`MDAnalysis` is used by RDF/MSD preprocessing scripts. Some preprocessing
scripts also depend on local WMI-MD postprocessing utilities that are not part
of this directory.

## Configuration

Edit `config.py` before running the pipeline.

| Name | Meaning |
| --- | --- |
| `DATA_ROOT` | Root directory containing preprocessed MD outputs, such as `shell_exchange.txt`, `pair_check/`, and MSD files. Defaults to `code/data`, or to `IL_DATA_ROOT` if the environment variable is set. |
| `RAW_DATA_ROOT` | Optional root directory for raw trajectories used by preprocessing scripts. Defaults to `code/raw_data`, or to `IL_RAW_DATA_ROOT` if set. |
| `FIGURE_ROOT` | Optional figure output root. Defaults to `../paper/Images`, or to `IL_FIGURE_ROOT` if set. |
| `ANIONS` | System names used by the scripts: `fsi`, `tfsi`, and `beti`. |
| `TEMPERATURES` | Temperatures analyzed by default: 298, 353, 373, and 423 K. |
| `RCUT` | Li-anion solvation-shell cutoff radius, in Angstrom. |
| `TABLE1` | Inter-event fit parameters and mean shell-change intervals used by the classifier. |
| `TABLE2_NSTART` | Hard-state offset used in selected pair-survival and burstiness calculations. |

For local data, set the environment variable before running the scripts:

```bash
export IL_DATA_ROOT=/path/to/preprocessed/md
```

If the preprocessing scripts are used, set the raw trajectory root as well:

```bash
export IL_RAW_DATA_ROOT=/path/to/raw/trajectories
```

Expected input paths include:

```text
DATA_ROOT/{anion}/{T}/shell_exchange.txt
DATA_ROOT/{anion}/{T}/pair_check/{ion_index}.txt
DATA_ROOT/code/diffusion/msd/{anion}/hard/{T}/1.0.txt
DATA_ROOT/code/diffusion/msd/{anion}/total/{T}.txt
```

## Running the Pipeline

Run commands from this directory:

```bash
cd path/to/il_paper/code
```

Run the full pipeline:

```bash
python run_all.py
```

Run one step:

```bash
python run_all.py --step theta
python run_all.py --step table4
python run_all.py --step table5
```

Run one anion:

```bash
python run_all.py --anion fsi
```

Skip slow steps when their outputs already exist:

```bash
python run_all.py --skip classify,table1
```

Valid step names are:

```text
classify
split
f_fit
collect
theta
table1
table2_alpha
table3
table4
table5
```

## Pipeline Order

```text
1. classify       classification/Echecker.py
                  shell_exchange.txt -> classification/result/

2. split          classification/soft_hard_split.py
                  classification/result/ -> classification/x/

3. f_fit          classification/f_pow.py
                  pair survival fits -> results/table2_f_powerlaw.csv

4. collect        analysis/table4/{duration,pair,shell_change,total}.py
                  event data -> classification/event_collect/

5. theta          analysis/table2/theta.py
                  soft-state fractions -> results/table2_theta.csv

6. table1         analysis/table1/inter.py
                  total shell-change inter-event statistics

7. table2_alpha   analysis/table2/h_fit.py
                  residence-time fits -> results/table2_h_exponential.csv

8. table3         analysis/table3/model_validate.py
                  synthetic validation of the clustering rule

9. table4         analysis/table4/burstiness.py
                  burstiness metrics -> results/table4_burstiness.csv

10. table5        analysis/table5/diffusion.py
                  state-resolved mobility estimates
```

`table3/model_validate.py` reads `config.TABLE1` and can be run independently
of most generated classification data. The diffusion step requires
`results/table2_theta.csv` and the MSD files listed in `DATA_ROOT`.

## Main Outputs

The main CSV outputs are written to `results/`.

| Output | Contents |
| --- | --- |
| `table1_inter_event.csv` | Total shell-change inter-event fit parameters and mean intervals. |
| `table2_f_powerlaw.csv` | State-resolved pair-survival power-law fit parameters. |
| `table2_h_exponential.csv` | Soft- and hard-state residence-time exponential fit parameters. |
| `table2_theta.csv` | Soft-state fraction for each anion and temperature. |
| `table4_burstiness.csv` | Burstiness parameter with bootstrap confidence intervals. |
| `table5_diffusion.csv` | Total, hard-state, and inferred soft-state mobility estimates. |
| `arrhenius.csv` | Temperature dependence of fitted state-transition parameters. |
| `ion_resolved_theta.csv` | Per-Li+ soft-state fractions. |
| `ion_resolved_alpha.csv` | Per-Li+ residence-time fit parameters. |
| `sensitivity_theta.csv` | Threshold-multiplier sensitivity of the soft-state fraction. |
| `sensitivity_alpha.csv` | Threshold-multiplier sensitivity of residence-time fit parameters. |

Some scripts append to existing CSV files. To regenerate a table from scratch,
remove the corresponding output file first.

## Soft/Hard State Classification

The classifier reads `shell_exchange.txt`, where each row is a time step and
each Li+ column indicates whether a shell-change event occurred.

The classification rule is:

1. Fit the total shell-change inter-event distribution and obtain the mean
   interval `delta_t`.
2. Group events separated by less than `delta_t` into candidate soft trains.
3. Merge candidate soft trains separated by less than `2 * delta_t`.
4. Label the merged intervals as soft state.
5. Label the remaining intervals as hard state.

The canonical threshold multiplier is 1.0. The code variable for this value is
named `TRAIL` in some scripts, and generated files use names such as
`1.0_0.txt`.

Generated state intervals are stored as:

```text
classification/result/{anion}/{soft|hard}/{T}/1.0_{ion_index}.txt
```

Each line contains one state interval. The first number is the starting frame,
and the last number is the ending frame.

Example:

```text
225 226
702 712
1186 1195 1211 1212 1229 1230
```

## Fitting Utilities

`classification/dist.py` contains the shared fitting functions.

| Function | Used for | Procedure |
| --- | --- | --- |
| `pow_calc(data)` | Pair-survival and shell-change inter-event distributions. | Binned log-log histogram followed by linear regression over the selected finite window. |
| `exp_calc(data)` | Auxiliary exponential fits. | Automatic binning with an x-shift before linear regression. |
| `h_calc(data, bins=100)` | Soft- and hard-state residence-time distributions. | Fixed 100-bin histogram without x-shift. |

Each function returns:

```text
(fx, fy, coeff, coeff_se, inter, inter_se, indic)
```

For new systems, inspect the resulting plots and fit windows before using the
parameters downstream.

## Mobility Estimates

`analysis/table5/diffusion.py` fits MSD curves by linear regression:

```text
D = slope / 6
```

`D_total` and `D_hard` are obtained from the corresponding MSD files.
`D_soft` is inferred from the soft-state fraction:

```text
D_soft = (D_total - (1 - theta) * D_hard) / theta
```

The MSD fitting windows are set manually in `HARD_RANGES` and `TOTAL_RANGES`
inside `analysis/table5/diffusion.py`.

## Additional Checks

The top-level helper scripts reproduce several supporting checks:

```bash
python run_arrhenius.py
python run_ion_resolved.py
python run_sensitivity.py
python run_bridging_timeline.py
```

These scripts write CSV files to `results/` and figures to directories under
`../paper/Images/`. They assume that the main classification outputs already
exist.

## Data Set Used in the Manuscript

The manuscript analysis used:

| Item | Value |
| --- | --- |
| Anions | FSI-, TFSI-, BETI- |
| Temperatures | 298, 353, 373, 423 K |
| Li+ ions per trajectory | 5 |
| Production trajectory length | 100 ns |
| Analysis stride | 1 ps |
| Analysis frames | 100,000 |

The current scripts are written around this data layout. New systems may require
updates to file paths, atom counts, fit windows, and the constants in
`config.py`.
