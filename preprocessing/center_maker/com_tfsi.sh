SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

for anion_model in tfsi
do
 for Ts in 298 353 373 423
 do
  export anion=$anion_model
  export T=$Ts
  sbatch -J ${anion_model}_${Ts} "${SCRIPT_DIR}/doslurm_tfsi.sh"
 done
done
