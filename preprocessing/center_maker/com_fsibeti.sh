SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RAW_ROOT="${IL_RAW_DATA_ROOT:-${SCRIPT_DIR}/../../raw_data}"

for anion_model in beti fsi
do
 for Ts in 298 353 373 423
 do
  export anion=$anion_model
  export T=$Ts
  cd "${RAW_ROOT}/wmi-md/NVT/${anion_model^^}/${T}"
  sbatch -J ${anion_model}_${Ts} "${SCRIPT_DIR}/doslurm_others.sh"
 done
done
