#!/bin/bash
#SBATCH --job-name=ising_ml
#SBATCH --account=cherngroup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${ISING_PYTHON:-python}" "$SLURM_SUBMIT_DIR/dynamical_benchmark.py" run --track ml --device cuda --trial "$SLURM_ARRAY_TASK_ID" "$@"
