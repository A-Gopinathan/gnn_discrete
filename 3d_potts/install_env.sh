#!/bin/bash
# Python dependencies; C++ ED additionally needs Armadillo (see README).
set -euo pipefail
ENV_NAME="${1:-potts_shared}"
CONDA="${CONDA_EXE:-conda}"
command -v "$CONDA" >/dev/null || { printf 'Conda must be available.\n' >&2; exit 1; }
[[ "$ENV_NAME" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || exit 1
BASE="$("$CONDA" info --base)"
ENVS="$("$CONDA" env list --json)"
"$BASE/bin/python" -c 'import json,os,sys
if sys.argv[1] in ("base","root") or any(os.path.basename(os.path.normpath(p))==sys.argv[1] for p in json.loads(sys.argv[2])["envs"]):
    raise SystemExit("Environment exists; activate it or choose a new name.")' "$ENV_NAME" "$ENVS"
"$CONDA" create --name "$ENV_NAME" python=3.11 pip -y
"$CONDA" run --name "$ENV_NAME" python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
"$CONDA" run --name "$ENV_NAME" python -m pip install numpy==2.4.6 matplotlib==3.10.8 torch-geometric==2.7.0
printf '\nActivate with: conda activate %s\n' "$ENV_NAME"
