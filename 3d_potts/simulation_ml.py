"""Batched independent random-site heat-bath chains; no energy surrogate history."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from local_gnn import PHYSICS, heatbath_sample, load_local, verify_full


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, writer):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def json_atomic(path, value):
    atomic(path, lambda stream: stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()))


def checkpoint_digest(saved):
    """Checksum logical content, independent of torch's serialization container."""
    def plain(value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items() if k != "checkpoint_sha256"}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    content = json.dumps(plain(saved), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def read_campaign(directory):
    from campaign_tools import verify
    c = verify(directory)
    if c.get("format") != "POTTS_DYNAMICS_V1":
        raise ValueError("invalid campaign format")
    for name in ("campaign_id", "model_sha256"):
        value = c.get(name, "")
        if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError(f"invalid {name}")
    for name in ("length", "sweeps", "chains"):
        if type(c.get(name)) is not int or c[name] < (0 if name == "sweeps" else 1):
            raise ValueError(f"invalid {name}")
    if c["length"] < 8 or c["length"] > 128 or c["length"] % 2 or c["chains"] > 100:
        raise ValueError("Require even L in [8,128] and chains<=100")
    for k, v in PHYSICS.items():
        if c.get(k) != v:
            raise ValueError(f"campaign physics mismatch: {k}")
    for key, value in dict(initial_seed_base=2026101100, ed_seed_base=2026201100,
                           ml_seed_base=2026301100).items():
        if c.get(key) != value:
            raise ValueError(f"invalid {key}")
    if c.get("model_checkpoint") != "model/best.pt":
        raise ValueError("campaign checkpoint must be model/best.pt")
    if sha256(directory / c["model_checkpoint"]) != c["model_sha256"]:
        raise ValueError("checkpoint SHA256 mismatch")
    return c


class Chain:
    def __init__(self, root, campaign_dir, campaign, trial, fast):
        self.trial, self.campaign = trial, campaign
        self.n = campaign["length"] ** 3
        self.directory = root / f"trial_{trial:03d}"
        initial_path = campaign_dir / "initial_states" / f"trial_{trial:03d}.bin"
        expected = campaign["initial_sha256"][initial_path.name]
        initial = initial_path.read_bytes()
        if len(initial) != self.n or hashlib.sha256(initial).hexdigest() != expected:
            raise ValueError(f"trial {trial}: initial state checksum/length mismatch")
        if max(initial) > 2:
            raise ValueError("initial state contains invalid states")
        self.identity = dict(format="POTTS_ML_CHAIN_V1", campaign_id=campaign["campaign_id"],
                             model_sha256=campaign["model_sha256"], initial_sha256=expected,
                             trial=trial, seed=campaign["ml_seed_base"] + trial,
                             length=campaign["length"], sweeps=campaign["sweeps"],
                             deterministic=not fast, physics=PHYSICS,
                             algorithm="sequential_random_site_heatbath", units="total electronic internal energy")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.raw = self.directory / "states.bin"
        self.checkpoint = self.directory / "checkpoint.pt"
        self.marker = self.directory / "COMPLETE"
        self.rng = np.random.Generator(np.random.PCG64(self.identity["seed"]))
        self.digest = hashlib.sha256()
        if self.checkpoint.exists():
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if saved.get("checkpoint_sha256") != checkpoint_digest(saved):
                raise ValueError("checkpoint content checksum mismatch")
            if saved["identity"] != self.identity or json.loads((self.directory / "metadata.json").read_text()) != self.identity:
                raise ValueError("resume identity mismatch")
            self.attempts = saved["attempts"]
            self.changed = saved["changed"]
            self.elapsed = saved["elapsed"]
            self.history = saved["history"]
            self.state = saved["state"].numpy().copy()
            self.last_completed = saved["last_completed"].numpy().copy()
            self.sites = saved["sites"]
            self.uniforms = saved["uniforms"]
            self.rng.bit_generator.state = saved["rng"]
            if (type(self.attempts) is not int or not 0 <= self.attempts <= self.n * campaign["sweeps"]
                    or not 0 <= self.changed <= self.attempts or not math.isfinite(self.elapsed) or self.elapsed < 0
                    or self.state.shape != (self.n,) or self.state.dtype != np.uint8 or np.any(self.state > 2)
                    or self.last_completed.shape != (self.n,) or self.last_completed.dtype != np.uint8
                    or np.any(self.last_completed > 2)):
                raise ValueError("invalid checkpoint state/counters")
            completed = self.attempts // self.n
            if len(self.history) != completed + 1 or any(row[0] != i or row[1] != i * self.n
                                                        for i, row in enumerate(self.history)):
                raise ValueError("invalid checkpoint progress history")
            if any(not 0 <= row[2] <= row[1] or not math.isfinite(row[3]) or row[3] < 0
                   for row in self.history):
                raise ValueError("invalid checkpoint progress values")
            offset = self.attempts % self.n
            if offset and (self.sites is None or self.uniforms is None):
                raise ValueError("missing unconsumed RNG draws")
            if self.sites is not None:
                if (self.sites.shape != (self.n,) or self.uniforms.shape != (self.n,)
                        or not np.issubdtype(self.sites.dtype, np.integer)
                        or np.any(self.sites < 0) or np.any(self.sites >= self.n)
                        or not np.isfinite(self.uniforms).all() or np.any(self.uniforms < 0)
                        or np.any(self.uniforms >= 1)):
                    raise ValueError("invalid buffered RNG draws")
            committed = (completed + 1) * self.n
            if not self.raw.exists() or self.raw.stat().st_size < committed:
                raise ValueError("missing/short raw trajectory")
            with open(self.raw, "rb") as stream:
                remaining = committed
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    self.digest.update(block)
                    remaining -= len(block)
                stream.seek(0)
                if stream.read(self.n) != initial:
                    raise ValueError("frame zero mismatch")
                stream.seek(completed * self.n)
                if stream.read(self.n) != self.last_completed.tobytes():
                    raise ValueError("last committed frame mismatch")
            if self.digest.hexdigest() != saved["raw_sha256"]:
                raise ValueError("committed raw trajectory corruption")
            if not offset and not np.array_equal(self.state, self.last_completed):
                raise ValueError("checkpoint state disagrees with completed frame")
            if self.marker.exists() and (not self.done or self.marker.read_text() != campaign["campaign_id"] + "\n"
                                         or self.raw.stat().st_size != committed):
                raise ValueError("stale/invalid COMPLETE marker")
            # Discard only frames beyond the atomic checkpoint's committed prefix.
            with open(self.raw, "r+b") as stream:
                stream.truncate(committed)
            self.write_progress()
        else:
            if any(self.directory.iterdir()):
                raise ValueError("trial output exists without a checkpoint; refusing overwrite")
            self.state = np.frombuffer(initial, dtype=np.uint8).copy()
            self.last_completed = self.state.copy()
            self.attempts, self.changed, self.elapsed = 0, 0, 0.0
            self.history = [[0, 0, 0, 0.0]]
            self.sites = self.uniforms = None
            atomic(self.raw, lambda stream: stream.write(initial))
            self.digest.update(initial)
            json_atomic(self.directory / "metadata.json", self.identity)
            self.save()
        self.started = time.monotonic()
        self.elapsed_base = self.elapsed

    @property
    def done(self):
        return self.attempts == self.n * self.campaign["sweeps"]

    def draws(self):
        if self.sites is None:
            self.sites = self.rng.integers(0, self.n, self.n, dtype=np.int64)
            self.uniforms = self.rng.random(self.n)
        offset = self.attempts % self.n
        return self.sites[offset], self.uniforms[offset]

    def write_progress(self):
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["sweep", "attempts", "changed_sites", "elapsed_seconds"])
        writer.writerows(self.history)
        atomic(self.directory / "progress.csv", lambda stream: stream.write(output.getvalue().encode()))

    def save(self):
        saved = dict(identity=self.identity, state=torch.from_numpy(self.state.copy()),
                     last_completed=torch.from_numpy(self.last_completed.copy()), attempts=self.attempts,
                     changed=self.changed, elapsed=self.elapsed, history=self.history,
                     sites=self.sites, uniforms=self.uniforms, rng=self.rng.bit_generator.state,
                     raw_sha256=self.digest.hexdigest())
        saved["checkpoint_sha256"] = checkpoint_digest(saved)
        atomic(self.checkpoint, lambda stream: torch.save(saved, stream))
        self.write_progress()
        if self.done:
            atomic(self.marker, lambda stream: stream.write((self.campaign["campaign_id"] + "\n").encode()))

    def sync(self, state, changed, completed_sweep=False):
        self.state = state
        self.changed = changed
        self.elapsed = self.elapsed_base + time.monotonic() - self.started
        if completed_sweep:
            frame = self.state.tobytes()
            with open(self.raw, "ab") as stream:
                stream.write(frame)
                stream.flush()
                os.fsync(stream.fileno())
            self.digest.update(frame)
            self.last_completed = self.state.copy()
            self.sites = self.uniforms = None
            self.history.append([self.attempts // self.n, self.attempts, self.changed, self.elapsed])
        self.save()


@contextlib.contextmanager
def trial_locks(root, trials):
    root.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for trial in sorted(trials):
            stream = stack.enter_context(open(root / f"trial_{trial:03d}.lock", "a+b"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"trial {trial} already locked") from None
        yield


@torch.inference_mode()
def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("ML evolution requires a Slurm allocation (SLURM_JOB_ID)")
    if (not math.isfinite(args.max_seconds) or args.max_seconds < 0 or args.stop_after_attempts < 0
            or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0):
        raise ValueError("invalid runtime/checkpoint limits")
    c = read_campaign(args.campaign)
    trials = list(range(c["chains"])) if args.trials is None else args.trials
    if not trials or len(set(trials)) != len(trials) or any(t < 0 or t >= c["chains"] for t in trials):
        raise ValueError("trials must be unique campaign indices")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(not args.fast)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    with trial_locks(args.output, trials):
        chains = [Chain(args.output, args.campaign, c, t, args.fast) for t in trials]
        active = [chain for chain in chains if not chain.done]
        if not active:
            for chain in chains:
                if not chain.marker.exists():
                    chain.save()
            return dict(status="complete", trials=trials, rounds=0)
        checkpoint = torch.load(args.campaign / c["model_checkpoint"], map_location="cpu", weights_only=False)
        model = load_local(checkpoint, c["length"], device)
        states = torch.as_tensor(np.stack([chain.state for chain in active]), device=device)
        changed = torch.tensor([chain.changed for chain in active], device=device, dtype=torch.int64)
        finite = torch.ones(len(active), device=device, dtype=torch.bool)
        # Validate on CPU: reference scatter on CUDA can be nondeterministic and
        # may require unavailable optional PyG extension binaries.
        reference_model = load_local(checkpoint, c["length"], "cpu")
        error = verify_full(reference_model, torch.from_numpy(active[0].state),
                            torch.tensor([0, c["length"] ** 3 - 1, c["length"] ** 3 // 2]))
        del reference_model, checkpoint
        stop = [False]
        old_handlers = {}
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        started = last_checkpoint = time.monotonic()
        rounds = 0
        try:
            while active:
                if stop[0] or (args.max_seconds and time.monotonic() - started >= args.max_seconds):
                    break
                if args.stop_after_attempts and rounds >= args.stop_after_attempts:
                    break
                draws = [chain.draws() for chain in active]
                sites = torch.tensor([d[0] for d in draws], device=device, dtype=torch.int64)
                uniforms = torch.tensor([d[1] for d in draws], device=device, dtype=torch.float64)
                delta = model(states, sites)
                finite.logical_and_(torch.isfinite(delta).all(-1))
                candidate = heatbath_sample(delta, uniforms, c["temperature"]).to(torch.uint8)
                row = torch.arange(len(active), device=device)
                changed += candidate != states[row, sites]
                states[row, sites] = candidate
                for chain in active:
                    chain.attempts += 1
                rounds += 1
                boundary = [chain.attempts % chain.n == 0 for chain in active]
                interval = time.monotonic() - last_checkpoint >= args.checkpoint_seconds
                if any(boundary) or interval:
                    if not bool(finite.all().cpu()):
                        raise FloatingPointError("nonfinite model predictions; previous checkpoints preserved")
                    host_states, host_changed = states.cpu().numpy(), changed.cpu().tolist()
                    for i, chain in enumerate(active):
                        if boundary[i] or interval:
                            chain.sync(host_states[i].copy(), host_changed[i], boundary[i])
                    if interval:
                        last_checkpoint = time.monotonic()
                        print(json.dumps(dict(status="running", rounds=rounds,
                                              wall_seconds=last_checkpoint - started,
                                              attempts={str(chain.trial): chain.attempts for chain in active},
                                              attempts_per_second=rounds / max(last_checkpoint - started, 1e-9))),
                              flush=True)
                    keep = [i for i, chain in enumerate(active) if not chain.done]
                    if len(keep) != len(active):
                        active = [active[i] for i in keep]
                        states, changed, finite = states[keep], changed[keep], finite[keep]
            if active:
                if not bool(finite.all().cpu()):
                    raise FloatingPointError("nonfinite predictions; previous checkpoints preserved")
                host_states, host_changed = states.cpu().numpy(), changed.cpu().tolist()
                for i, chain in enumerate(active):
                    chain.sync(host_states[i].copy(), host_changed[i])
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        return dict(status="complete" if all(chain.done for chain in chains) else "paused", trials=trials,
                    rounds=rounds, wall_seconds=time.monotonic() - started, reference_max_abs_error=error,
                    attempts={str(chain.trial): chain.attempts for chain in chains},
                    changed_sites={str(chain.trial): chain.changed for chain in chains})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--stop-after-attempts", type=int, default=0,
                        help="pause after this many per-chain rounds in this invocation; zero is unlimited")
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    parser.add_argument("--fast", action="store_true", help="explicitly allow nondeterministic torch kernels")
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "complete" else 75
    except (ValueError, RuntimeError, OSError, KeyError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
