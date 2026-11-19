#!/bin/bash
#SBATCH --job-name=potts_correlations
#SBATCH --account=cherngroup
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=potts_correlations_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec "${POTTS_PYTHON:-python}" "$SLURM_SUBMIT_DIR/dynamical_benchmark.py" plot "$@"
