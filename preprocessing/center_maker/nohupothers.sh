#!/bin/bash
echo $T
echo $anion
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python3 -u "${SCRIPT_DIR}/fsi_beti.py" ${T} ${anion}
