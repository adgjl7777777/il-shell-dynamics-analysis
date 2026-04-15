"""
Master pipeline runner for the IL paper analysis.

Runs all steps required to reproduce Tables 1-5 from MD trajectory data.
Each step writes to results/ as a CSV file.

EXECUTION ORDER:
  Step 1  classify      — Echecker.py: classify soft/hard states (long, ~10 min per anion)
  Step 2  split         — soft_hard_split.py: sort pairs into x11/x12/x21/x22/loyal
  Step 3  f_fit         — f_pow.py: power-law β(f) ± SE → results/table2_f_powerlaw.csv
  Step 4  collect       — duration/pair/shell_change/total: event data for burstiness
  Step 5  theta         — theta.py: soft-state fraction θ → results/table2_theta.csv
  Step 6  table1        — inter.py: inter-event β, Δt → results/table1_inter_event.csv
  Step 7  table2_alpha  — h_fit.py: exponential α(h) ± SE → results/table2_h_exponential.csv
  Step 8  table4        — burstiness.py: A_N ± 95% CI → results/table4_burstiness.csv
  Step 9  table5        — diffusion.py: D coefficients → results/table5_diffusion.csv

NOTE: Step 1 requires DATA_ROOT (shell_exchange.txt files).
      Step 6 (table1/inter.py) also reads DATA_ROOT directly and is slow.
      Steps 4–9 require Steps 1–3 to have completed first.
      table3 (model_validate.py) only needs config.TABLE1 and can run standalone.

Usage:
    python run_all.py                        # run all steps
    python run_all.py --step classify        # run one step only
    python run_all.py --anion fsi            # run all steps for one anion
    python run_all.py --skip classify,table1 # skip slow steps if data exists
"""
import subprocess
import sys
import os

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
ANIONS    = ["fsi", "tfsi", "beti"]
TEMPS     = ["298", "353", "373", "423"]

# Directories (run scripts from their own directory so relative paths work)
CLASSIFY  = os.path.join(CODE_ROOT, "classification")
TABLE1    = os.path.join(CODE_ROOT, "analysis", "table1")
TABLE2    = os.path.join(CODE_ROOT, "analysis", "table2")
TABLE3    = os.path.join(CODE_ROOT, "analysis", "table3")
TABLE4    = os.path.join(CODE_ROOT, "analysis", "table4")
TABLE5    = os.path.join(CODE_ROOT, "analysis", "table5")


PYTHON = sys.executable  # 현재 실행 중인 python 인터프리터 경로 사용


def run(cmd, cwd):
    # "python" → 현재 인터프리터로 교체
    cmd = [PYTHON if c == "python" else c for c in cmd]
    print(f"  $ cd {os.path.relpath(cwd, CODE_ROOT)} && {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def step_classify(anions):
    """Echecker: classify each anion's trajectories into soft/hard (slow)."""
    for anion in anions:
        run(["python", "Echecker.py", anion], CLASSIFY)


def step_split(anions):
    """soft_hard_split: sort pair trajectories into x11/x12/x21/x22/loyal."""
    for anion in anions:
        run(["python", "soft_hard_split.py", anion], CLASSIFY)


def step_f_fit(anions):
    """f_pow: power-law β(f) ± SE → results/table2_f_powerlaw.csv."""
    for anion in anions:
        run(["python", "f_pow.py", anion], CLASSIFY)


def step_collect(anions):
    """Collect event data for burstiness (duration, pair, shell_change, total)."""
    for anion in anions:
        run(["python", "duration.py",     anion], TABLE4)
    for anion in anions:
        run(["python", "pair.py",         anion], TABLE4)
    for anion in anions:
        run(["python", "shell_change.py", anion], TABLE4)
    run(["python", "total.py"], TABLE4)  # loops over all anions internally


def step_theta(anions=None):
    """theta: soft-state fraction θ → results/table2_theta.csv."""
    run(["python", "theta.py"], TABLE2)


def step_table1(anions):
    """inter: inter-event P(τ) → results/table1_inter_event.csv (slow)."""
    for anion in anions:
        for T in TEMPS:
            run(["python", "inter.py", anion, T], TABLE1)


def step_table2_alpha(anions):
    """h_fit: exponential α(h) ± SE → results/table2_h_exponential.csv."""
    for anion in anions:
        run(["python", "h_fit.py", anion], TABLE2)


def step_table3(anions):
    """model_validate: theoretical h(n) model (standalone, reads config only)."""
    for anion in anions:
        for i in range(4):
            run(["python", "model_validate.py", anion, str(i)], TABLE3)


def step_table4(anions):
    """burstiness: A_N ± 95% CI → results/table4_burstiness.csv."""
    for anion in anions:
        run(["python", "burstiness.py", anion], TABLE4)


def step_table5(anions=None):
    """diffusion: D coefficients → results/table5_diffusion.csv."""
    run(["python", "diffusion.py"], TABLE5)


STEPS = {
    "classify":     step_classify,
    "split":        step_split,
    "f_fit":        step_f_fit,
    "collect":      step_collect,
    "theta":        step_theta,
    "table1":       step_table1,
    "table2_alpha": step_table2_alpha,
    "table3":       step_table3,
    "table4":       step_table4,
    "table5":       step_table5,
}

ORDERED = ["classify", "split", "f_fit", "collect", "theta",
           "table1", "table2_alpha", "table3", "table4", "table5"]


def parse_args():
    args  = sys.argv[1:]
    step  = args[args.index("--step")  + 1] if "--step"  in args else None
    anion = args[args.index("--anion") + 1] if "--anion" in args else None
    skip_str = args[args.index("--skip") + 1] if "--skip" in args else ""
    skip  = set(skip_str.split(",")) if skip_str else set()
    return step, ([anion] if anion else ANIONS), skip


if __name__ == "__main__":
    step_arg, anions, skip = parse_args()
    run_steps = [step_arg] if step_arg else ORDERED

    for name in run_steps:
        if name in skip:
            print(f"\n--- Skipping: {name} ---")
            continue
        print(f"\n{'='*50}")
        print(f"=== Step: {name} ({'all anions' if not step_arg else ', '.join(anions)}) ===")
        print(f"{'='*50}")
        fn = STEPS[name]
        # theta and table5 loop internally; pass anions for steps that need it
        import inspect
        sig = inspect.signature(fn)
        if list(sig.parameters.keys()) == [] or list(sig.parameters.keys()) == ['anions'] and sig.parameters['anions'].default is None:
            fn(anions)
        else:
            fn(anions)

    print("\n=== Done. Results in results/ ===")
    for f in sorted(os.listdir(os.path.join(CODE_ROOT, "results"))):
        if f.endswith(".csv"):
            path = os.path.join(CODE_ROOT, "results", f)
            with open(path) as fh:
                n = sum(1 for _ in fh) - 1
            print(f"  {f:40s}  {n} rows")
