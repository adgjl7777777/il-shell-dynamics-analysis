SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "${SCRIPT_DIR}/out"

for anion_model in beti
do
 for Ts in 298 353 373 423
 do
  export anion=$anion_model
  export T=$Ts
  nohup bash "${SCRIPT_DIR}/nohupothers.sh" > "${SCRIPT_DIR}/out/${anion_model}_${Ts}_nohup.out" 2>&1 &
  sleep 1s
 done
done
