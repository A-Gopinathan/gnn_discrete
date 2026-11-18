# Ising Workflow

Run these commands from the repository root. The model is a 10-layer,
width-64 residual GNN with a final-layer readout, not multiscale aggregation.

## 1. Install the environment

With Conda available, `install_env.sh` creates an isolated Python 3.11 environment
and installs PyTorch (CUDA 11.8 build), NumPy and Matplotlib:

```bash
bash install_env.sh
conda activate ising_shared
export ISING_PYTHON="$(command -v python)"
```

## 2. Use the completed dataset

Data generation is finished: 64 L26 chains, 256 sweeps, 24 labeled snapshots per
chain, with a fixed 48/8/8 training/validation/test split. Nothing needs regenerating.

```bash
DATASET=/sfs/gpfs/tardis/home/kaz9hp/GNN_Lattice_Models/2d_ising/production/runs/L26_s256_n64_20261118/data/pooled
```

This is the original cluster location. If the dataset is copied elsewhere,
change `DATASET` to its new directory; no data files are bundled here.

## 3. Train

`models.py` defines the network; `train.py` performs training; `train.slurm`
requests an A100 GPU.

```bash
TRAIN=$(sbatch --parsable train.slurm --data "$DATASET")
squeue --all -j "$TRAIN"
```

Training creates `trained_best.pt` (selected by validation MSE) and
`trained_last.pt` (for resuming). Wait for the job to finish successfully:

```bash
sacct -j "$TRAIN" --format=JobID,State,ExitCode
MODEL="$PWD/trained_best.pt"
```

## 4. Run the benchmarks

`static_benchmark.py` produces validation parity and numerical metrics;
`static.slurm` runs it on an A100:

```bash
sbatch static.slurm --data "$DATASET" --checkpoint "$MODEL"
```

Outputs are `parity_validation.png`, `.pdf` and `.npz`. Use `--test` only for the
final held-out evaluation after model choices are fixed; it blocks further training.

`dynamical_benchmark.py` runs eight matched ED/GNN pairs on L32 for 1000 sweeps.
`physics.py` supplies ED energies and Glauber probabilities. ED uses CPUs;
GNN inference uses A100s. The comparison waits for both arrays to finish:

```bash
ED=$(sbatch --parsable ed.slurm)
ML=$(sbatch --parsable ml.slurm --checkpoint "$MODEL")
sbatch --dependency=afterok:$ED:$ML compare.slurm
```

Trajectories go to `benchmark/`; comparison produces `radial.png`, `radial.pdf`
and `radial.npz`, including paired-chain uncertainties.

## 5. Generate large-system images

`large_scale.py` runs one ML-only L50 trajectory for 1000 sweeps. Once it finishes,
`plot_domains.py` plots the saved configurations:

```bash
sbatch large_scale.slurm --checkpoint "$MODEL"
# After that job completes:
python plot_domains.py
```

Results go to `large_scale/` and `domains.png/.pdf`. This is an illustration,
not an independent ED accuracy benchmark.

For interrupted jobs, resubmit the same arguments with `--resume` after checking
that no copy is running. Simulation exit 75 means checkpointed but unfinished;
resubmit failed array IDs and update dependent job IDs before comparison.
For options, use actual filenames, e.g. `python train.py --help` or
`python dynamical_benchmark.py run --help`.
