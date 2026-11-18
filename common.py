"""Small shared helpers; settings and restart state live inside data/checkpoints."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from physics import ENERGY_CONVENTION, Physics, require_slurm

ROOT = Path(__file__).resolve().parent
DATA = ROOT
COMPACT_DATA_SHA256 = "e287a6f861ca82325af242b8dee094f2e5d3075fcf278325834b7f44f06bd8b4"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path, **arrays):
    with atomic_file(path) as stream:
        np.savez_compressed(stream, **arrays)


def atomic_torch_save(path, payload):
    import torch
    with atomic_file(path) as stream:
        torch.save(payload, stream)


def read_split(split, path):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    path = Path(path).resolve()
    if path.is_dir():
        digest = sha256(path / "manifest.json")
        manifest = json.loads((path / "manifest.json").read_text())
        complete = json.loads((path / "COMPLETE").read_text())
        if complete != {"identity": manifest["identity"], "manifest_sha256": digest}:
            raise ValueError("Incomplete or changed dataset")
        records = {row["trial"]: row for row in manifest["trials"]}
        if len(records) != 64 or len(manifest["trials"]) != 64:
            raise ValueError("Expected the 64-chain production dataset")
        ids = sorted(records)
        random.Random(2026111805).shuffle(ids)
        splits = dict(train=sorted(ids[:48]), validation=sorted(ids[48:56]), test=sorted(ids[56:]))
        groups = {key: [] for key in ("spins", "delta", "trial", "sweep")}
        for trial in splits[split]:
            file = (path / records[trial]["file"]).resolve()
            if not file.is_relative_to(path) or sha256(file) != records[trial]["sha256"]:
                raise ValueError("Invalid data file or checksum")
            with np.load(file, allow_pickle=False) as saved:
                if int(saved["trial"]) != trial:
                    raise ValueError("Wrong chain ID")
                groups["spins"].append(saved["spins"])
                groups["delta"].append(saved["delta"])
                groups["trial"].append(np.full(len(saved["spins"]), trial))
                groups["sweep"].append(saved["sweeps"])
        result = {key: np.concatenate(value) for key, value in groups.items()}
        result.update(all_splits=splits, source_manifest_sha256=digest,
                      physics=manifest["physics"], energy_convention=manifest["energy_convention"])
    else:
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            result = {key: saved[f"{split}_{key}"].copy() for key in ("spins", "delta", "trial", "sweep")}
            result["all_splits"] = {key: saved[f"{key}_ids"].tolist() for key in ("train", "validation", "test")}
            result["source_manifest_sha256"] = str(saved["source_manifest_sha256"].item())
            result["physics"] = {key: float(saved[key]) for key in ("temperature", "filling", "hopping")}
            result["energy_convention"] = str(saved["energy_convention"].item())
    result["manifest_verified"] = path.is_dir()
    result.update(split_ids=result["all_splits"][split], data_sha256=digest)
    spins, delta = result["spins"], result["delta"]
    if (spins.ndim != 3 or spins.shape != delta.shape or not np.isin(spins, [-1, 1]).all()
            or not np.isfinite(delta).all() or spins.shape[1] != spins.shape[2]
            or result["trial"].shape != (len(spins),) or result["sweep"].shape != (len(spins),)
            or set(result["trial"]) != set(result["split_ids"])):
        raise ValueError("Invalid split arrays")
    ids = sum(result["all_splits"].values(), [])
    if len(set(ids)) != len(ids) or any(not group for group in result["all_splits"].values()):
        raise ValueError("Chain splits overlap or are empty")
    Physics(**result["physics"])
    if result["energy_convention"] != ENERGY_CONVENTION:
        raise ValueError("Unexpected label convention")
    return result


def load_model(path, device):
    from models import load_checkpoint
    model, checkpoint = load_checkpoint(path, device)
    if checkpoint["physics"] != {"temperature": .01, "filling": .5, "hopping": 1.0}:
        raise ValueError("Checkpoint physics mismatch")
    if checkpoint.get("energy_convention") != ENERGY_CONVENTION:
        raise ValueError("Checkpoint energy convention mismatch")
    return model, checkpoint


def validate_checkpoint_data(checkpoint, data):
    if "data_sha256" in checkpoint:
        valid = checkpoint["data_sha256"] == data["data_sha256"]
    else:
        valid = ((data["manifest_verified"] or data["data_sha256"] == COMPACT_DATA_SHA256) and
                 checkpoint.get("dataset_manifest_sha256") == data["source_manifest_sha256"])
    if not valid or checkpoint["splits"] != data["all_splits"] or checkpoint["physics"] != data["physics"]:
        raise ValueError("Checkpoint and dataset/splits do not match")


def training_allowed():
    if (DATA / ".test_exposed").exists():
        raise ValueError("Test set already evaluated; further fitting is blocked")


def expose_test():
    DATA.mkdir(exist_ok=True)
    (DATA / ".test_exposed").touch(exist_ok=True)
