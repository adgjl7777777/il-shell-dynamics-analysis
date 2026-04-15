#!/bin/sh
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
#python3 -u "${SCRIPT_DIR}/unwrap.py" ${anions} ${Ts}
#python3 -u "${SCRIPT_DIR}/plot.py" ${anions} ${Ts}
python3 -u "${SCRIPT_DIR}/diffusion.py" ${anions} ${Ts}
