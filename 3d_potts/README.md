# 3D Potts Workflow

Run commands from this folder. The model is already trained: **3 layers, width
64, selected epoch 197**. The next large job is GNN-driven **L50 dynamics**, not
training a model or generating ED data at L50. Do not refit the test-exposed baseline.

## Install and locate the model

```bash
bash install_env.sh
conda activate potts_shared
export POTTS_PYTHON="$(command -v python)"
MODEL=/sfs/gpfs/tardis/home/kaz9hp/potts_gnn_runs/baseline_20260928T232058Z_z8vR6U/training/best.pt
PROJECT=/sfs/gpfs/tardis/home/kaz9hp/GNN_Lattice_Models/3d_potts
```

An existing compatible `gnn_env` can be used instead. These are original-cluster
input paths; change them if files are copied elsewhere. No weights or data are
uploaded with the code. Slurm defaults to `cherngroup`; override the account if needed.

## Start the L50 simulation

```bash
sbatch large_scale.slurm --checkpoint "$MODEL" --output "$PWD/runs/L50"
```

`large_scale.py` uses the existing selected-site model, with three independently
seeded trajectories batched on one A100. Each runs 1000 sweeps, with **125,000
random site selections per sweep**. This is a long job; saved checkpoints allow
continuation with the same command plus `--resume`. Exit 75 means saved but unfinished.
Do not launch two writers for the same run. The L12 ordering time does not
establish the L50 ordering time, and a large image is not an ED accuracy test.

After saved frames exist, use `plot_lattice.py` for fixed-slice snapshots and
a late-time 3D dumbbell inset. It verifies committed frames before plotting.

```bash
python plot_lattice.py --campaign "$PWD/runs/L50" \
  --trial-dir "$PWD/runs/L50/ml/trial_000" --sweeps 0,100,1000 \
  --slice-axis z --slice-index 25 --inset-origin 23,23,23 --inset-size 4 \
  --output-prefix L50
```

## Static benchmark

`static_benchmark.py` plots only nonself transitions, with MAE/MSE/RMSE in
physical total-energy units. It restores the checkpoint scale automatically.
To reproduce the existing held-out results without new inference:

```bash
python static_benchmark.py \
  --predictions "$PROJECT/static_benchmark/results/baseline_epoch197/test_predictions.npz" \
  --stages --output-prefix parity
```

For fresh model evaluation on the prepared dataset:

```bash
sbatch static.slurm --checkpoint "$MODEL" \
  --dataset "$PROJECT/data_three/prepared_potts_L10_T0p10" --output-prefix parity_new
```

## Dynamical benchmark

The main plots show x/y/z correlations separately: ED blue connected curves,
GNN red open squares without connecting lines, and SEM across independent chains.
The existing report tables can be replotted without rerunning simulations:

```bash
python dynamical_benchmark.py plot \
  --table "$PROJECT/data_three/report_11_11/dynamical_bench/per_chain_correlations.csv" \
  --output "$PWD/runs/report_replot"
```

Those saved report cohorts contain 30 pairs through sweep 400 and 25 at sweep 800.
They are not relabeled using newer completion counts. Full numerical values are
retained; the default report-style nonnegative y range warns if values are hidden.

To generate a new matched L12 reference and GNN campaign, first build the C++ ED
engine (unnecessary for the L50 GNN job):

```bash
module load gcc/14.2.0 openmpi/5.0.7 armadillo/15.0.1
make
python dynamical_benchmark.py prepare --checkpoint "$MODEL" \
  --output "$PWD/runs/L12" --length 12 --sweeps 1000 --chains 30
ED=$(sbatch --parsable ed.slurm --output "$PWD/runs/L12")
ML=$(sbatch --parsable ml.slurm --output "$PWD/runs/L12")
sbatch --dependency=afterok:$ED:$ML compare.slurm --output "$PWD/runs/L12"
```

The native C++ physics and checkpointing sources are included. New runs keep
their own frozen source, weights and initial states; original campaign metadata
and scratch paths are never rewritten. Use fixed matched cohorts for new plots.

## Conventions

States 0/1/2 mean X/Y/Z. Parameters are T=0.10, filling=0.5, alpha=1,
beta=-0.5, gamma=-0.7, with periodic boundaries and site order `x*L*L+y*L+z`.
The energy is finite-temperature fermionic **internal energy**, with no classical
term and no division by site count. Three-state heat bath includes the current
state and selects L^3 sites with replacement per sweep. The inference scale is
loaded from the checkpoint (baseline: 0.7374496106286567).
