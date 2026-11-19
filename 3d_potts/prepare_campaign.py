"""Prepare an immutable POTTS_DYNAMICS_V1 campaign; never submit or run jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCES = ["simulation_ed.cpp", "simulation_ml.py", "local_gnn.py", "potts_model.hpp",
           "potts_model.cpp", "util.hpp", "util.cpp", "potts_gnn.py", "Makefile",
           "prepare_campaign.py", "campaign_tools.py", "large_scale.py", "dynamical_benchmark.py"]
# Preserved baseline campaign 8f720bda3b5978612896db1fd0a2eb9653ab47b5c4d6a17afea11ed0040438b3.
# These match the training source snapshot; no external training directory is needed.
SCIENCE_SHA256 = {
    "potts_gnn.py": "a79219951778c854cceaf6249b2966cbfc7799d9a2310178f1155d84afc333f6",
    "potts_model.cpp": "688df4916fd207394b541eb3339693b7daa9239ab27321f39ef532bfd028b888",
    "potts_model.hpp": "562f02269da13eb8f2fc6208dc23f6430aff7f0876ea24e048b913d31925d43b",
    "util.cpp": "ac6e171f504470c82061ff47f4665d662e7d1f5fb5b165270700468f60ca6e78",
    "util.hpp": "3a90c47963420102baccc8d1d2b408d3aa9531e7852c4d3c23588dcbc9ac6d75",
}
BASELINE_SHA256 = "3552f7e37f55ec23836f4242c7b8eed96a228b9d45325096172c3727ce0e1eae"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_current_sources(metadata):
    """Reject running a frozen campaign with changed entrypoints or engines."""
    if set(metadata["source_sha256"]) != set(SOURCES):
        raise ValueError("Campaign source set differs; use its original frozen workflow")
    for name, digest in metadata["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Current source differs from frozen campaign: " + name)


def prepare(output, checkpoint, length=12, sweeps=1000, chains=30, backup_root=None):
    """Create a campaign, or verify an identical existing one without rewriting it."""
    from campaign_tools import verify

    destination = Path(output).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if (not 4 <= length <= 128 or length % 2 or not 0 <= sweeps <= 100000
            or not 1 <= chains <= 100):
        raise ValueError("Require even L in [4,128], 0<=sweeps<=100000, 1<=chains<=100")
    if destination == ROOT or (ROOT in destination.parents and not destination.is_relative_to(ROOT / "runs")):
        raise ValueError("Use --output under runs/ or outside the source folder")
    backup_path = (Path(backup_root).expanduser().resolve() if backup_root is not None
                   else destination / "backup")
    if destination.exists():
        metadata = verify(destination)
        verify_current_sources(metadata)
        expected = dict(length=length, sweeps=sweeps, chains=chains,
                        model_sha256=sha256(checkpoint_path))
        if backup_root is not None:
            expected["backup_root"] = str(backup_path)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing campaign settings/checkpoint differ; refusing overwrite")
        return metadata
    for name in SOURCES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
    for name, digest in SCIENCE_SHA256.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Source differs from preserved baseline physics/model: " + name)
    if sha256(checkpoint_path) != BASELINE_SHA256:
        raise ValueError("Expected the tested epoch-197 baseline best.pt checkpoint")
    import torch
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    physics = config["metadata"]["physics"]
    expected = dict(temperature=0.1, filling=0.5, alpha=1.0, beta=-0.5, gamma=-0.7,
                    classical_energy=False, per_site_normalization=False,
                    energy="finite-temperature fermionic internal energy", boundary="periodic",
                    index_order="x*L*L+y*L+z", orbitals=["X", "Y", "Z"])
    if any(physics.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint physics differs from the fixed benchmark")
    if config["settings"].get("smoke") or config["settings"].get("overfit"):
        raise ValueError("Use a trained production checkpoint")
    if (config["model"] != dict(num_layers=3, hidden_dim=64) or saved["epoch"] != 197
            or saved["target_scale"] != 0.7374496106286567):
        raise ValueError("Expected baseline epoch 197, three layers, width 64 and training RMS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".tmp-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        for child in ("source", "initial_states", "model"):
            (folder / child).mkdir()
        for name in SOURCES:
            shutil.copy2(ROOT / name, folder / "source" / name)
        shutil.copy2(checkpoint_path, folder / "model/best.pt")
        metadata = dict(format="POTTS_DYNAMICS_V1", length=length, sweeps=sweeps, chains=chains,
                        **{key: expected[key] for key in ("temperature", "filling", "alpha", "beta", "gamma")},
                        initial_seed_base=2026101100, ed_seed_base=2026201100, ml_seed_base=2026301100,
                        physics=physics, model_checkpoint="model/best.pt",
                        model_sha256=sha256(folder / "model/best.pt"), model_config=config["model"],
                        checkpoint_epoch=saved["epoch"], target_scale=saved["target_scale"],
                        training_fingerprint=config["fingerprint"], numpy_version=np.__version__,
                        initial_rng="numpy.PCG64", paired_initial_conditions=True,
                        frame_layout="uint8[(completed_sweeps+1),L,L,L], C-order x*L*L+y*L+z; frame0=initial",
                        update_rule="random sequential three-state heat bath, L^3 sites with replacement per sweep",
                        backup_root=str(backup_path),
                        source_sha256={name: sha256(folder / "source" / name) for name in SOURCES})
        initial_hashes = {}
        for trial in range(chains):
            generator = np.random.Generator(np.random.PCG64(metadata["initial_seed_base"] + trial))
            states = generator.integers(0, 3, size=length ** 3, dtype=np.uint8)
            name = f"trial_{trial:03d}.bin"
            (folder / "initial_states" / name).write_bytes(states.tobytes())
            initial_hashes[name] = sha256(folder / "initial_states" / name)
        metadata["initial_sha256"] = initial_hashes
        metadata["campaign_id"] = fingerprint(metadata)
        (folder / "campaign.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        flat_keys = ("format", "campaign_id", "length", "sweeps", "chains", "temperature", "filling",
                     "alpha", "beta", "gamma", "initial_seed_base", "ed_seed_base", "ml_seed_base")
        (folder / "campaign.cfg").write_text("".join(f"{key}={metadata[key]}\n" for key in flat_keys))
        verify(folder)
        os.rename(folder, destination)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=30)
    parser.add_argument("--backup-root", type=Path)
    args = parser.parse_args(argv)
    metadata = prepare(**vars(args))
    print(f"Verified prepared campaign {metadata['campaign_id']}: {args.output.resolve()}")
    print("Paired initial conditions only; no simulations submitted.")


if __name__ == "__main__":
    main()
