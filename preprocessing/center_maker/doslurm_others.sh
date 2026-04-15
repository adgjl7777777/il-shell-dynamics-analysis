#!/bin/bash
#SBATCH -p gpupart
#SBATCH -o out/%x.o.%j.out
#SBATCH -e out/%x.e.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --nodelist=nanode04
#SBATCH -t 100000:00:00

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
###python -u "${SCRIPT_DIR}/fsi_beti.py"
python -u "${SCRIPT_DIR}/save.py"
