#!/bin/bash
# Create an isolated environment; never replace an existing one.
set -euo pipefail
ENV_NAME="${1:-ising_shared}"
CONDA="${CONDA_EXE:-conda}"
command -v "$CONDA" >/dev/null || { printf 'Conda must be available in this shell.\n' >&2; exit 1; }
[[ "$ENV_NAME" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]] || { printf 'Invalid environment name.\n' >&2; exit 1; }
BASE="$("$CONDA" info --base)"
ENVS="$("$CONDA" env list --json)"
"$BASE/bin/python" -c 'import json,os,sys
name=sys.argv[1]
if name in ("base", "root") or any(os.path.basename(os.path.normpath(p)) == name for p in json.loads(sys.argv[2])["envs"]):
    raise SystemExit("Environment already exists; activate it or choose another name.")' "$ENV_NAME" "$ENVS"
"$CONDA" create --name "$ENV_NAME" python=3.11 pip -y
"$CONDA" run --name "$ENV_NAME" python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
"$CONDA" run --name "$ENV_NAME" python -m pip install numpy==2.4.6 matplotlib==3.10.8
"$CONDA" run --name "$ENV_NAME" python -c 'import torch,numpy,matplotlib; print("Installed:",torch.__version__,numpy.__version__,matplotlib.__version__)'
printf '\nActivate with: conda activate %s\n' "$ENV_NAME"
