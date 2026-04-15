#!/bin/bash
#SBATCH -p intelpart
#SBATCH -o out/%x.o.%j.out
#SBATCH -e out/%x.e.%j.err
#SBATCH -N 1
#SBATCH -n 1
##SBATCH --nodelist=node12
#SBATCH -t 100000:00:00

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python -u "${SCRIPT_DIR}/save_tfsi.py"
###python -u "${SCRIPT_DIR}/save.py"
