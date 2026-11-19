# GNN Discrete Lattice Models

Code for the two parts of the project:

- [`2d_ising/`](2d_ising/): double-exchange Ising data loading, training, static
  parity, matched dynamics, and an L50 GNN simulation.
- [`3d_potts/`](3d_potts/): the established three-layer Potts model, static parity,
  directional correlations, ED reference code, and an L50 GNN simulation.

Each folder has its own README, installer and Slurm jobs. Run commands from the
corresponding model folder. Model weights, datasets and generated figures are
external inputs/outputs, not stored in this repository.

The Potts L50 job is a simulation using the existing trained checkpoint, not
training a new model on a 50 x 50 x 50 lattice. The Ising and Potts physics,
energy conventions, state encodings and correlation definitions are different;
the two workflows intentionally keep them separate.
