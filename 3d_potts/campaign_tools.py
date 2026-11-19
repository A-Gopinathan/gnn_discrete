"""Read-only campaign validation and locked, non-destructive result backups."""
import argparse
import csv
import fcntl
import json
import os
import platform
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from prepare_campaign import fingerprint, sha256


def verify(campaign):
    campaign = Path(campaign).resolve()
    metadata = json.loads((campaign / "campaign.json").read_text())
    payload = dict(metadata)
    saved = payload.pop("campaign_id")
    if fingerprint(payload) != saved:
        raise ValueError("Campaign metadata fingerprint mismatch")
    rows = (campaign / "campaign.cfg").read_text().splitlines()
    cfg = dict(row.split("=", 1) for row in rows)
    if len(cfg) != len(rows) or any(str(metadata[key]) != value for key, value in cfg.items()):
        raise ValueError("Campaign JSON/config disagreement")
    required = {"format", "campaign_id", "length", "sweeps", "chains", "temperature", "filling", "alpha",
                "beta", "gamma", "initial_seed_base", "ed_seed_base", "ml_seed_base"}
    if set(cfg) != required or cfg["format"] != "POTTS_DYNAMICS_V1":
        raise ValueError("Invalid campaign configuration")
    for name, digest in metadata["source_sha256"].items():
        if Path(name).name != name or sha256(campaign / "source" / name) != digest:
            raise ValueError("Frozen source modified: " + name)
    for name, digest in metadata["initial_sha256"].items():
        path = campaign / "initial_states" / name
        if Path(name).name != name or sha256(path) != digest:
            raise ValueError("Initial state checksum mismatch: " + name)
        states = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        if len(states) != metadata["length"] ** 3 or np.any(states > 2):
            raise ValueError("Invalid initial lattice: " + name)
    if sha256(campaign / metadata["model_checkpoint"]) != metadata["model_sha256"]:
        raise ValueError("Model checkpoint checksum mismatch")
    return metadata


def backup(campaign, root, track, trials):
    metadata = verify(campaign)
    destination = Path(metadata["backup_root"]) / (campaign.name + "_" + metadata["campaign_id"][:12])
    (destination / ("results_" + track)).mkdir(parents=True, exist_ok=True)
    for trial in trials:
        name = f"trial_{trial:03d}"
        source = root / name
        if not source.is_dir():
            continue
        lockpath = root / (name + ".lock")
        # Do not copy a moving checkpoint/raw-file transaction.
        with lockpath.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = destination / ("results_" + track) / name
            target.mkdir(parents=True, exist_ok=True)
            with (target / ".backup.lock").open("a") as backup_lock:
                fcntl.flock(backup_lock, fcntl.LOCK_EX)
                generations = target / "generations"
                generations.mkdir(exist_ok=True)
                # Never replace individual files in the only recovery copy. An
                # interrupted copy leaves the previous complete generation intact.
                with tempfile.TemporaryDirectory(prefix=".staging-", dir=generations) as temporary:
                    staging = Path(temporary)
                    for path in source.iterdir():
                        if path.is_file() and not path.name.endswith(".tmp"):
                            copied = staging / path.name
                            shutil.copy2(path, copied)
                            with copied.open("rb") as handle:
                                os.fsync(handle.fileno())
                    snapshot = generations / ("snapshot_" + str(time.time_ns()))
                    os.rename(staging, snapshot)
                for directory in (snapshot, generations):
                    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                link = target / (".latest-" + str(os.getpid()))
                link.symlink_to(snapshot.relative_to(target), target_is_directory=True)
                os.replace(link, target / "latest")
                descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    print("Backed up available trials to " + str(destination / ("results_" + track)))
    runtime = root / ("runtime_" + os.environ.get("SLURM_JOB_ID", "manual") + ".json")
    if runtime.is_file():
        shutil.copy2(runtime, destination / ("results_" + track) / runtime.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "backup", "status", "runtime"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--track", choices=("ed", "ml"))
    parser.add_argument("--trials", type=int, nargs="+")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    metadata = verify(campaign)
    if args.action == "runtime":
        if args.output is None or args.track is None:
            parser.error("runtime requires --output and --track")
        args.output.mkdir(parents=True, exist_ok=True)
        record = {"track": args.track, "hostname": platform.node(), "python": platform.python_version(),
                  "campaign_id": metadata["campaign_id"],
                  "environment": {key: os.environ.get(key) for key in
                                  ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "LOADEDMODULES",
                                   "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        if args.track == "ml":
            import torch
            record.update(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
        path = args.output / ("runtime_" + os.environ.get("SLURM_JOB_ID", "manual") + ".json")
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    elif args.action == "status":
        benchmark = campaign.parent.parent
        for track in ("ed", "ml"):
            complete = 0
            for trial in range(metadata["chains"]):
                root = args.output / track if args.output else campaign / track
                if not root.exists() and args.output is None:
                    root = benchmark / ("results_" + track) / campaign.name
                path = root / f"trial_{trial:03d}"
                sweep = -1
                if (path / "progress.csv").is_file():
                    with (path / "progress.csv").open() as handle:
                        rows = list(csv.DictReader(handle))
                    if rows:
                        sweep = int(rows[-1]["sweep"])
                done = (path / "COMPLETE").is_file()
                complete += done
                print(f"{track.upper()} trial_{trial:03d}: completed_sweep={sweep} COMPLETE={done}")
            print(f"{track.upper()}: {complete}/{metadata['chains']} completion markers (not an integrity audit)")
    elif args.action == "backup":
        if args.output is None or args.track is None:
            parser.error("backup requires --output and --track")
        trials = args.trials if args.trials is not None else list(range(metadata["chains"]))
        if any(t < 0 or t >= metadata["chains"] for t in trials):
            parser.error("Invalid trial ID")
        backup(campaign, args.output.resolve(), args.track, trials)
    else:
        print("Verified campaign " + metadata["campaign_id"])


if __name__ == "__main__":
    main()
