#!/usr/bin/env python3
"""Inventory candidate input files for the R2-1 power-law/model-comparison revision task.

This script reads existing old-project outputs but writes only under the revision folder.
It does not modify manuscript, old code, or old data files.
"""
from pathlib import Path
import csv
import numpy as np

OLD_CODE = Path("/nas_2/transcendence/il_paper/code")
REV_DIR = Path("/nas_2/transcendence/revision")
OUT_DIR = REV_DIR / "analysis" / "powerlaw_model_comparison" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = REV / "notes" / "01_r2_1_distribution_statistics" / "analysis_output_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = ["298", "353", "373", "423"]

rows = []

def summarize(path: Path, dataset: str, anion: str, temp: str, state: str):
    try:
        data = np.loadtxt(path)
    except Exception as exc:
        rows.append({
            "dataset": dataset, "anion": anion, "T": temp, "state": state,
            "path": str(path), "exists": path.exists(), "n": 0,
            "min": "", "median": "", "mean": "", "p95": "", "max": "",
            "note": f"read_error: {exc}",
        })
        return
    arr = np.asarray(data, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        note = "empty_after_positive_filter"
        stats = {k: "" for k in ["min", "median", "mean", "p95", "max"]}
    else:
        note = "duration_values_ps_or_steps"
        stats = {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }
    rows.append({
        "dataset": dataset, "anion": anion, "T": temp, "state": state,
        "path": str(path), "exists": path.exists(), "n": int(arr.size),
        **stats, "note": note,
    })

# Best candidate for state-resolved pair survival f_soft/f_hard: already duration-only files.
for anion in ANIONS:
    for temp in TEMPS:
        for state in ["soft", "hard"]:
            p = OLD_CODE / "classification" / "event_collect" / "event#2(Pair_breaking;survival)" / state / anion / temp / "data.txt"
            summarize(p, "pair_survival_event_collect", anion, temp, state)

# Candidate for shell-change inter-event distributions.
for anion in ANIONS:
    for temp in TEMPS:
        p = OLD_CODE / "classification" / "event_collect" / "event#1(Shell_change;interevent)" / "total" / anion / temp / "total.txt"
        summarize(p, "shell_change_interevent_total", anion, temp, "total")
        for state in ["soft", "hard"]:
            p = OLD_CODE / "classification" / "event_collect" / "event#1(Shell_change;interevent)" / state / anion / temp / "data.txt"
            summarize(p, "shell_change_interevent_state", anion, temp, state)

# Candidate for state residence h_soft/h_hard.
for anion in ANIONS:
    for temp in TEMPS:
        for state in ["soft", "hard"]:
            p = OLD_CODE / "classification" / "event_collect" / "soft_hard_duration" / state / anion / temp / "data.txt"
            summarize(p, "state_residence_duration", anion, temp, state)

out_csv = OUT_DIR / "input_inventory.csv"
with out_csv.open("w", newline="") as f:
    fieldnames = ["dataset", "anion", "T", "state", "exists", "n", "min", "median", "mean", "p95", "max", "path", "note"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

summary_md = SUMMARY_DIR / "input_inventory_summary.md"
with summary_md.open("w") as f:
    f.write("# Power-law/model-comparison input inventory\n\n")
    f.write("This inventory was generated from existing old-project files. No old files were modified.\n\n")
    f.write(f"CSV: `{out_csv}`\n\n")
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        ok = [r for r in subset if r["exists"] and r["n"] > 0]
        f.write(f"## {dataset}\n\n")
        f.write(f"Files with positive data: {len(ok)} / {len(subset)}\n\n")
        for r in ok[:8]:
            f.write(f"- {r["anion"]} {r["T"]} K {r["state"]}: n={r["n"]}, median={r["median"]:.3g}, p95={r["p95"]:.3g}, max={r["max"]:.3g}\n")
        if len(ok) > 8:
            f.write(f"- ... {len(ok)-8} more rows in CSV.\n")
        f.write("\n")

print(out_csv)
print(summary_md)
