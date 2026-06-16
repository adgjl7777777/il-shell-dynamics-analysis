
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np

BASE = Path("/nas_2/transcendence/_delete/cowork/my_work")
OUT_DIR = Path("/nas_2/transcendence/revision/analysis/uncertainty_bootstrap/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIONS = ["fsi", "tfsi", "beti"]
TEMPS = [298, 353, 373, 423]
N_IONS = 5
N_BLOCKS = 5
N_BOOT = 20000
RNG = np.random.default_rng(20260530)


def parse_pair_file(path: Path):
    frames = []
    with path.open() as fh:
        for line in fh:
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if not parts:
                continue
            vals = [int(x) for x in parts]
            frames.append(set(vals[1:]))
    return frames


def load_system(anion: str, T: int):
    files = [BASE / anion / str(T) / "pair_check" / f"{ion}.txt" for ion in range(N_IONS)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing pair_check files: " + "; ".join(missing))
    per_ion = [parse_pair_file(f) for f in files]
    n = min(len(x) for x in per_ion)
    return [x[:n] for x in per_ion], n


def frame_metrics(shells):
    counts = {}
    for ion, ids in enumerate(shells):
        for aid in ids:
            counts.setdefault(aid, []).append(ion)
    bridge_anions = {aid: ions for aid, ions in counts.items() if len(ions) >= 2}

    pair_flags = []
    for i, j in itertools.combinations(range(N_IONS), 2):
        pair_flags.append(1.0 if shells[i].intersection(shells[j]) else 0.0)

    li_with_bridge = set()
    for ions in bridge_anions.values():
        li_with_bridge.update(ions)

    return (
        1.0 if bridge_anions else 0.0,
        float(len(bridge_anions)),
        float(np.mean(pair_flags)),
        len(li_with_bridge) / N_IONS,
        float(max((len(v) for v in bridge_anions.values()), default=0)),
    )


def bootstrap_block_mean(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(means.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_system(anion: str, T: int):
    per_ion, n_frames = load_system(anion, T)
    block_size = n_frames // N_BLOCKS
    block_rows = []
    all_sums = np.zeros(5, dtype=float)
    all_count = 0

    for b in range(N_BLOCKS):
        start = b * block_size
        stop = n_frames if b == N_BLOCKS - 1 else (b + 1) * block_size
        sums = np.zeros(5, dtype=float)
        for k in range(start, stop):
            shells = [per_ion[ion][k] for ion in range(N_IONS)]
            sums += np.asarray(frame_metrics(shells), dtype=float)
        count = stop - start
        means = sums / count
        all_sums += sums
        all_count += count
        block_rows.append({
            "anion": anion,
            "T": T,
            "block": b,
            "n_frames": count,
            "any_bridge_fraction": means[0],
            "mean_bridging_anions_per_frame": means[1],
            "pair_shared_fraction_mean": means[2],
            "li_bridge_fraction_mean": means[3],
            "max_bridge_multiplicity_mean": means[4],
        })

    means = all_sums / all_count
    any_blocks = [r["any_bridge_fraction"] for r in block_rows]
    pair_blocks = [r["pair_shared_fraction_mean"] for r in block_rows]
    li_blocks = [r["li_bridge_fraction_mean"] for r in block_rows]
    any_boot = bootstrap_block_mean(any_blocks)
    pair_boot = bootstrap_block_mean(pair_blocks)
    li_boot = bootstrap_block_mean(li_blocks)

    return {
        "anion": anion,
        "T": T,
        "n_frames": all_count,
        "any_bridge_fraction": means[0],
        "any_bridge_block_boot_se": any_boot[0],
        "any_bridge_block_boot_lo95": any_boot[1],
        "any_bridge_block_boot_hi95": any_boot[2],
        "any_bridge_block_min": min(any_blocks),
        "any_bridge_block_max": max(any_blocks),
        "mean_bridging_anions_per_frame": means[1],
        "pair_shared_fraction_mean": means[2],
        "pair_shared_block_boot_se": pair_boot[0],
        "pair_shared_block_boot_lo95": pair_boot[1],
        "pair_shared_block_boot_hi95": pair_boot[2],
        "pair_shared_block_min": min(pair_blocks),
        "pair_shared_block_max": max(pair_blocks),
        "li_bridge_fraction_mean": means[3],
        "li_bridge_block_boot_se": li_boot[0],
        "li_bridge_block_boot_lo95": li_boot[1],
        "li_bridge_block_boot_hi95": li_boot[2],
        "li_bridge_block_min": min(li_blocks),
        "li_bridge_block_max": max(li_blocks),
        "max_bridge_multiplicity_mean": means[4],
        "max_bridge_multiplicity_max_block_mean": max(r["max_bridge_multiplicity_mean"] for r in block_rows),
    }, block_rows


def write_csv(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    summaries = []
    blocks = []
    for anion in ANIONS:
        for T in TEMPS:
            summary, block_rows = summarize_system(anion, T)
            summaries.append(summary)
            blocks.extend(block_rows)

    summary_csv = OUT_DIR / "bridging_fraction_summary.csv"
    block_csv = OUT_DIR / "bridging_fraction_blocks.csv"
    write_csv(summary_csv, summaries)
    write_csv(block_csv, blocks)

    print("Wrote " + str(summary_csv))
    print("Wrote " + str(block_csv))
    print("Highest any-bridge fractions:")
    for r in sorted(summaries, key=lambda x: x["any_bridge_fraction"], reverse=True)[:5]:
        print("  {anion} {T} K any={any:.3f} pair={pair:.3f} li={li:.3f}".format(
            anion=r["anion"].upper(),
            T=r["T"],
            any=r["any_bridge_fraction"],
            pair=r["pair_shared_fraction_mean"],
            li=r["li_bridge_fraction_mean"],
        ))
    b = next(r for r in summaries if r["anion"] == "beti" and r["T"] == 298)
    print("BETI 298: any={any:.3f} block={lo:.3f}-{hi:.3f} pair={pair:.3f} li={li:.3f}".format(
        any=b["any_bridge_fraction"],
        lo=b["any_bridge_block_min"],
        hi=b["any_bridge_block_max"],
        pair=b["pair_shared_fraction_mean"],
        li=b["li_bridge_fraction_mean"],
    ))


if __name__ == "__main__":
    main()
