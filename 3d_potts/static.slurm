#!/bin/bash
#SBATCH --job-name=potts_static
#SBATCH --account=cherngroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=potts_static_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec "${POTTS_PYTHON:-python}" "$SLURM_SUBMIT_DIR/static_benchmark.py" --device cuda "$@"
