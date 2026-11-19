#!/bin/bash
#SBATCH --job-name=potts_ed
#SBATCH --account=cherngroup
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=potts_ed_%A_%a.log
set -euo pipefail
module load gcc/14.2.0 openmpi/5.0.7 armadillo/15.0.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "${POTTS_PYTHON:-python}" "$SLURM_SUBMIT_DIR/dynamical_benchmark.py" run --track ed --trial "$SLURM_ARRAY_TASK_ID" "$@"
