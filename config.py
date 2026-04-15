"""
Shared configuration constants for the IL paper analysis pipeline.

Import in any script with:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import ANIONS, TEMPERATURES, DATA_ROOT, RCUT, TABLE1, TABLE2_NSTART
    from config import CODE_ROOT, CLASSIFY_DIR, RESULTS_DIR

Directory layout (all relative to CODE_ROOT = this file's directory):
    preprocessing/   - MD preprocessing (COM centering, shell_exchange)
    classification/  - Echecker + soft/hard split; outputs to classification/result/
    analysis/        - Table computations (table1 through table5)
    auxiliary/       - CN, RDF
    results/         - Final CSV outputs (one file per paper table)
"""
import os as _os
CODE_ROOT    = _os.path.dirname(_os.path.abspath(__file__))
CLASSIFY_DIR = _os.path.join(CODE_ROOT, "classification")
RESULTS_DIR  = _os.path.join(CODE_ROOT, "results")

# ── System parameters ────────────────────────────────────────────────────────
ANIONS       = ["fsi", "tfsi", "beti"]
TEMPERATURES = [298, 353, 373, 423]   # K

# ── Data paths ───────────────────────────────────────────────────────────────
# Preprocessed MD-derived files are stored outside this repository by default.
# Set IL_DATA_ROOT before running the analysis on real trajectory data.
DATA_ROOT = _os.environ.get("IL_DATA_ROOT", _os.path.join(CODE_ROOT, "data"))

# Optional raw trajectory root used by preprocessing scripts.
RAW_DATA_ROOT = _os.environ.get("IL_RAW_DATA_ROOT", _os.path.join(CODE_ROOT, "raw_data"))

# Figure output root used by selected plotting helpers.
FIGURE_ROOT = _os.environ.get(
    "IL_FIGURE_ROOT",
    _os.path.abspath(_os.path.join(CODE_ROOT, "..", "paper", "Images")),
)

# ── Solvation-shell cutoff radius (Å) ────────────────────────────────────────
# Determined from the first minimum of the Li-anion RDF (~5.5 Å)
RCUT = {
    "fsi":  5.5,
    "tfsi": 5.5,
    "beti": 5.5,
}

# ── Table 1: Inter-event time distribution parameters ───────────────────────
# Format: TABLE1[anion][T] = (n_threshold, beta, delta_t_exp)
#   n_threshold : truncation point of the power-law fit (manually determined)
#   beta        : power-law exponent β  → loaded from results/table1_inter_event.csv
#   delta_t_exp : mean inter-event time Δt (ps)  → loaded from results/table1_inter_event.csv
_N_THRESHOLD = {
    "fsi":  {298: 900,   353: 100,  373: 100,  423: 40},
    "tfsi": {298: 10000, 353: 1000, 373: 500,  423: 200},
    "beti": {298: 20000, 353: 2000, 373: 2000, 423: 200},
}

def _load_table1():
    import csv as _csv
    _path = _os.path.join(RESULTS_DIR, "table1_inter_event.csv")
    _t1 = {}
    if _os.path.exists(_path):
        with open(_path) as _fh:
            for _row in _csv.DictReader(_fh):
                _a, _T = _row["anion"], int(_row["T"])
                _t1.setdefault(_a, {})[_T] = (
                    _N_THRESHOLD[_a][_T],
                    float(_row["beta"]),
                    float(_row["mean_tau_ps"]),
                )
    else:
        # Fallback to hardcoded values if CSV not yet generated
        _t1 = {
            "fsi":  {298: (900,   1.6532, 30.65), 353: (100,  1.8383, 18.91),
                     373: (100,   1.9371, 16.20),  423: (40,   2.0438, 12.44)},
            "tfsi": {298: (10000, 1.4363, 131.30), 353: (1000, 1.3998, 71.66),
                     373: (500,   1.4449, 53.86),  423: (200,  1.4956, 33.96)},
            "beti": {298: (20000, 1.4450, 155.27), 353: (2000, 1.3919, 85.28),
                     373: (2000,  1.4781, 62.28),  423: (200,  1.5819, 40.41)},
        }
    return _t1

TABLE1 = _load_table1()

# ── Table 2: Hard-state pair survival n_start offsets ────────────────────────
# n_start is the offset applied to hard-state pair survival times before power-law
# fitting (models the onset of relevant dynamics beyond the initial transient).
# Values read from Table 2 of the paper; used by f_hard burstiness calculation.
TABLE2_NSTART = {
    "fsi":  {298: 33.3,  353: 20.0,  373: 17.7,  423: 13.5},
    "tfsi": {298: 159.4, 353: 80.9,  373: 59.1,  423: 36.9},
    "beti": {298: 178.6, 353: 93.4,  373: 68.8,  423: 43.9},
}
