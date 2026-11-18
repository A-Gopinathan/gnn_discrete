#!/bin/bash
#SBATCH --job-name=ising_compare
#SBATCH --account=cherngroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec "${ISING_PYTHON:-python}" "$SLURM_SUBMIT_DIR/dynamical_benchmark.py" compare "$@"
