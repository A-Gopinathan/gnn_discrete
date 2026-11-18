#!/bin/bash
#SBATCH --job-name=ising_L50
#SBATCH --account=cherngroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${ISING_PYTHON:-python}" "$SLURM_SUBMIT_DIR/large_scale.py" --device cuda "$@"
