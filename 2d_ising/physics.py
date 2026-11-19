"""Projected itinerant Ising physics, with a separate classical bath temperature.

Sites are row-major (x * L + y), with periodic boundaries. The fermions are
spinless, canonical, half filled and at zero electronic temperature. There is
no classical spin-energy term. A flip delta always means E(new) - E(old).
"""

from dataclasses import dataclass
import os

import numpy as np


ENERGY_CONVENTION = "zero_temperature_half_filled_total_energy"
DELTA_CONVENTION = "Enew-Eold"


@dataclass(frozen=True)
class Physics:
    temperature: float = 0.01
    filling: float = 0.5
    hopping: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive")
        if self.filling != 0.5:
            raise ValueError("only canonical half filling (filling=0.5) is supported")
        if not np.isfinite(self.hopping) or self.hopping <= 0:
            raise ValueError("hopping must be finite and strictly positive")


def require_slurm():
    """Prevent accidental ED, simulation or training work on a login node."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Compute work requires a Slurm allocation (SLURM_JOB_ID)")


def validate_length(length):
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
        raise ValueError("length must be an even integer >= 4")
    if length < 4 or length % 2:
        raise ValueError("length must be an even integer >= 4")
    return int(length)


def validate_spins(spins):
    spins = np.asarray(spins)
    if spins.ndim != 2 or spins.shape[0] != spins.shape[1]:
        raise ValueError("spins must be a square [L,L] array")
    validate_length(spins.shape[0])
    if spins.dtype.kind not in "if" or not np.all((spins == -1) | (spins == 1)):
        raise ValueError("spins must contain signed -1/+1 values only")
    return spins


def hamiltonian(spins, physics=Physics()):
    """H_ij = -t (1+s_i*s_j)/2 on periodic nearest-neighbor bonds."""
    spins = validate_spins(spins)
    length = spins.shape[0]
    sites = np.arange(spins.size).reshape(length, length)
    matrix = np.zeros((spins.size, spins.size), dtype=np.float64)
    for axis in (0, 1):
        neighbors = np.roll(sites, -1, axis=axis)
        values = -physics.hopping * (spins == np.roll(spins, -1, axis=axis))
        matrix[sites.ravel(), neighbors.ravel()] = values.ravel()
        matrix[neighbors.ravel(), sites.ravel()] = values.ravel()
    return matrix


def energy(spins, physics=Physics()):
    """Return the zero-electron-T half-filled TOTAL energy, not energy/site."""
    spins = validate_spins(spins)
    require_slurm()
    eigenvalues = np.linalg.eigvalsh(hamiltonian(spins, physics))
    return float(np.sum(eigenvalues[: spins.size // 2], dtype=np.float64))


def flip_delta(spins, site, physics=Physics()):
    """Return E(flipped) - E(original), without modifying the input."""
    spins = validate_spins(spins)
    if isinstance(site, (bool, np.bool_)) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be a flat integer index")
    if not 0 <= site < spins.size:
        raise ValueError("site is outside the lattice")
    flipped = spins.copy()
    flipped.flat[site] *= -1
    return energy(flipped, physics) - energy(spins, physics)


def glauber_probability(delta, temp):
    """Stable sigmoid(-delta/temp), for a scalar or an array of flip energies."""
    if not np.isscalar(temp) or not np.isfinite(temp) or temp <= 0:
        raise ValueError("temp must be finite and strictly positive")
    values = np.asarray(delta, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("delta must be finite")
    with np.errstate(over="ignore", under="ignore"):
        small = np.exp(-np.abs(values) / temp)
    result = np.where(values >= 0, small / (1 + small), 1 / (1 + small))
    return float(result) if result.ndim == 0 else result
