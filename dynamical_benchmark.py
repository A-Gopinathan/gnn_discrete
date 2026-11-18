"""Matched ED/GNN Glauber trajectories and raw radial correlations."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from common import DATA, ROOT, atomic_npz, load_model, lock, require_slurm, sha256
from physics import ENERGY_CONVENTION, Physics, energy, glauber_probability, validate_length


def source_hashes(track):
    names = ["dynamical_benchmark.py", "physics.py", "common.py"]
    if track == "ml":
        names.append("models.py")
    return {name: sha256(ROOT / name) for name in names}


def initial_spins(length, seed, trial):
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial, 0]))
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))


def read_trajectory(path, *, complete=False):
    """Validate saved state without ED, model inference, or pickle loading."""
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    settings = json.loads(str(data["settings"].item()))
    for key in ("attempts", "seed", "trial", "target_sweeps"):
        if data[key].shape != () or data[key].dtype.kind not in "iu":
            raise ValueError(f"Invalid integer {key}: {path}")
    length = validate_length(settings["length"])
    track, seed, trial = settings["track"], settings["seed"], settings["trial"]
    target, attempts = settings["target_sweeps"], int(data["attempts"])
    if (track not in {"ed", "ml"} or type(seed) is not int or seed < 0
            or type(trial) is not int or trial < 0 or type(target) is not int or target < 0):
        raise ValueError(f"Invalid trajectory settings: {path}")
    if (settings["physics"] != asdict(Physics())
            or settings["energy_convention"] != ENERGY_CONVENTION
            or settings["kernel"] != "glauber"
            or settings["sweep_convention"] != "N selections with replacement"
            or settings["source_sha256"] != source_hashes(track)):
        raise ValueError(f"Physics or source mismatch: {path}")
    digest = settings["checkpoint_sha256"]
    if (track == "ed" and digest != "") or (track == "ml" and
            (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest))):
        raise ValueError(f"Invalid checkpoint hash: {path}")
    for key in ("seed", "trial", "target_sweeps"):
        if int(data[key]) != settings[key]:
            raise ValueError(f"Inconsistent {key}: {path}")
    if not np.array_equal(data["initial_seed"], [seed, trial, 0]):
        raise ValueError(f"Invalid initialization seed: {path}")
    n = length * length
    if not 0 <= attempts <= target * n:
        raise ValueError(f"Invalid attempt count: {path}")
    times = np.arange(attempts // n + 1)
    frames, state = data["spins"], data["state"]
    if (not np.array_equal(data["saved_sweeps"], times)
            or frames.shape != (len(times), length, length) or state.shape != (length, length)
            or not np.isin(frames, [-1, 1]).all() or not np.isin(state, [-1, 1]).all()
            or not np.array_equal(frames[0], initial_spins(length, seed, trial))
            or (attempts % n == 0 and not np.array_equal(state, frames[-1]))):
        raise ValueError(f"Invalid frames or state: {path}")
    if data["complete"].dtype != np.dtype(bool) or data["complete"].shape != ():
        raise ValueError(f"Invalid completion flag: {path}")
    if bool(data["complete"]) != (attempts == target * n) or (complete and not bool(data["complete"])):
        raise ValueError(f"Trajectory is incomplete or completion flag is inconsistent: {path}")
    if track == "ed" and not np.isfinite(float(data["current_energy"])):
        raise ValueError(f"Invalid cached ED energy: {path}")
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(str(data["rng_state"].item()))
    data["settings"] = settings
    return data


def run(args):
    """Run one chain; return 0 when complete, 75 after an unfinished checkpoint."""
    require_slurm()
    length = validate_length(args.length)
    if args.track not in {"ed", "ml"} or min(args.seed, args.trial, args.sweeps) < 0:
        raise ValueError("Invalid track, seed, trial, or sweep count")
    if args.max_attempts is not None and args.max_attempts < 0:
        raise ValueError("--max-attempts must be nonnegative")
    if args.track == "ml" and args.checkpoint is None:
        raise ValueError("ML dynamics requires an explicit --checkpoint")
    physics, predictor = Physics(), None
    checkpoint_hash = sha256(args.checkpoint) if args.track == "ml" else ""
    if args.track == "ml":
        import torch
        from models import SelectedSitePredictor
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, checkpoint = load_model(args.checkpoint, args.device)
        if sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed while loading")
        predictor = SelectedSitePredictor(model, length, checkpoint["target_scale"])
    settings = dict(track=args.track, length=length, seed=args.seed, trial=args.trial,
                    target_sweeps=args.sweeps, physics=asdict(physics),
                    energy_convention=ENERGY_CONVENTION, kernel="glauber",
                    sweep_convention="N selections with replacement",
                    checkpoint_sha256=checkpoint_hash, source_sha256=source_hashes(args.track),
                    numpy_version=np.__version__, torch_version=str(torch.__version__) if predictor else "",
                    device=args.device if args.track == "ml" else "cpu")
    output = Path(args.output_dir) / f"{args.track}_{args.trial:03d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    with lock(output.with_suffix(".lock")):
        if args.resume and not output.exists():
            raise FileNotFoundError(f"No trajectory to resume: {output}")
        previous = {}
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            handler = signal.getsignal(sig)
            if handler is not None:
                previous[sig] = handler
                signal.signal(sig, stop)
        try:
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, args.trial, 1 if args.track == "ed" else 2]))
            if output.exists():
                if not args.resume:
                    raise FileExistsError(f"{output} exists; use --resume")
                saved = read_trajectory(output)
                if saved["settings"] != settings:
                    raise ValueError("Resume requires identical settings, checkpoint, and source")
                state, frames = saved["state"].copy(), list(saved["spins"])
                attempts, current_energy = int(saved["attempts"]), float(saved["current_energy"])
                rng.bit_generator.state = json.loads(str(saved["rng_state"].item()))
                if bool(saved["complete"]):
                    print(f"Already complete: {output}")
                    return 0
            else:
                state = initial_spins(length, args.seed, args.trial)
                frames, attempts = [state.copy()], 0
                current_energy = energy(state, physics) if args.track == "ed" else np.nan
            n, spent = length * length, 0

            def save():
                atomic_npz(output, state=state, spins=np.asarray(frames, dtype=np.int8),
                           saved_sweeps=np.arange(len(frames)), attempts=attempts,
                           target_sweeps=args.sweeps, complete=attempts == args.sweeps * n,
                           current_energy=current_energy, rng_state=json.dumps(rng.bit_generator.state),
                           seed=args.seed, initial_seed=np.array([args.seed, args.trial, 0], dtype=np.uint64),
                           trial=args.trial, settings=json.dumps(settings, sort_keys=True))

            save()
            while attempts < args.sweeps * n and not stopped:
                if args.max_attempts is not None and spent >= args.max_attempts:
                    break
                site = int(rng.integers(n))
                if args.track == "ed":
                    candidate = state.copy()
                    candidate.flat[site] *= -1
                    candidate_energy = energy(candidate, physics)
                    delta = candidate_energy - current_energy
                else:
                    delta = float(predictor(state[None], np.array([site])).item())
                if rng.random() < glauber_probability(delta, physics.temperature):
                    state.flat[site] *= -1
                    if args.track == "ed":
                        current_energy = candidate_energy
                attempts += 1
                spent += 1
                if attempts % n == 0:
                    frames.append(state.copy())
                    save()
            save()
            finished = attempts == args.sweeps * n
            print(f"{'Complete' if finished else 'Checkpointed'}: {output} ({attempts}/{args.sweeps * n} attempts)")
            return 0 if finished else 75
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def radial_correlation(spins):
    """Raw periodic <s_i s_(i+d)>, averaged over equal-distance shells."""
    length = spins.shape[-1]
    displacement = np.minimum(np.arange(length), length - np.arange(length))
    squared = displacement[:, None] ** 2 + displacement[None, :] ** 2
    shells = np.unique(squared[squared <= (length // 2) ** 2])
    spectrum = np.fft.fft2(spins, axes=(-2, -1))
    correlation = np.fft.ifft2(spectrum * spectrum.conj(), axes=(-2, -1)).real / length ** 2
    return np.sqrt(shells), np.stack([correlation[..., squared == shell].mean(axis=-1)
                                     for shell in shells], axis=-1)


def sweep_list(value):
    sweeps = [int(item) for item in value.split(",")]
    if not sweeps or min(sweeps) < 0 or sweeps != sorted(set(sweeps)):
        raise ValueError("Sweeps must be nonnegative, unique, and increasing")
    return sweeps


def compare(args):
    if args.pairs < 2:
        raise ValueError("At least two independent pairs are needed for SEM")
    sweeps, records, reference, ml_hash = sweep_list(args.sweeps), [], None, None
    target = None if args.target_sweeps == "auto" else int(args.target_sweeps)
    for trial in range(args.pairs):
        pair = []
        for track in ("ed", "ml"):
            path = Path(args.input_dir) / f"{track}_{trial:03d}.npz"
            data = read_trajectory(path, complete=True)
            settings = data["settings"]
            if settings["trial"] != trial or settings["track"] != track:
                raise ValueError(f"Wrong trial or track in {path}")
            identity = {key: settings[key] for key in ("length", "seed", "target_sweeps", "physics")}
            if reference is None:
                reference = identity
                target = settings["target_sweeps"] if target is None else target
            if identity != reference or settings["target_sweeps"] != target or max(sweeps) > target:
                raise ValueError("All requested pairs must share length, seed, physics, and full target")
            if track == "ml":
                ml_hash = settings["checkpoint_sha256"] if ml_hash is None else ml_hash
                if settings["checkpoint_sha256"] != ml_hash:
                    raise ValueError("Mixed ML checkpoint weights")
            pair.append(data["spins"][sweeps])
            if track == "ed":
                initial = data["spins"][0]
            elif not np.array_equal(initial, data["spins"][0]):
                raise ValueError("ED/ML initial states are not matched")
        records.append(pair)
    radii, values = radial_correlation(np.asarray(records))
    ed, ml = values[:, 0], values[:, 1]
    difference = ml - ed
    sem = lambda values: values.std(axis=0, ddof=1) / np.sqrt(args.pairs)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(Path(str(prefix) + ".npz"), radii=radii, sweeps=sweeps,
               trial_ids=np.arange(args.pairs), ed=ed, ml=ml, difference=difference,
               ed_mean=ed.mean(0), ml_mean=ml.mean(0), difference_mean=difference.mean(0),
               ed_sem=sem(ed), ml_sem=sem(ml), difference_sem=sem(difference),
               settings=json.dumps(reference, sort_keys=True), checkpoint_sha256=ml_hash,
               observable="raw periodic equal-distance minimum-image shell correlation")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
    columns = min(4, len(sweeps))
    blocks = (len(sweeps) + columns - 1) // columns
    fig, axes = plt.subplots(2 * blocks, columns, figsize=(3.1 * columns, 4.4 * blocks),
                             squeeze=False, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1] * blocks})
    for index, sweep in enumerate(sweeps):
        row, col = divmod(index, columns)
        top, bottom = axes[2 * row, col], axes[2 * row + 1, col]
        top.errorbar(radii, ed.mean(0)[index], yerr=sem(ed)[index], color="tab:blue",
                     linewidth=1, label="ED")
        top.errorbar(radii, ml.mean(0)[index], yerr=sem(ml)[index], color="tab:red",
                     marker="s", markerfacecolor="none", markersize=3, linestyle="none", label="GNN")
        top.set(title=f"Sweep {sweep}", ylabel=r"$C(r)$")
        if index == 0:
            top.legend(frameon=False, fontsize=9)
        bottom.axhline(0, color="0.6", linewidth=0.7)
        bottom.errorbar(radii, difference.mean(0)[index], yerr=sem(difference)[index],
                        color="tab:red", marker="s", markerfacecolor="none", markersize=3,
                        linestyle="none")
        bottom.set(xlabel=r"$r$", ylabel=r"$C_{\rm GNN}-C_{\rm ED}$")
    for index in range(len(sweeps), blocks * columns):
        row, col = divmod(index, columns)
        axes[2 * row, col].set_visible(False)
        axes[2 * row + 1, col].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(str(prefix) + "." + suffix, dpi=220)
    plt.close(fig)
    print(f"Compared all {args.pairs} complete matched pairs: {prefix}.npz/.png/.pdf")
    return 0


def add_run_arguments(parser, *, track=True):
    if track:
        parser.add_argument("--track", choices=("ed", "ml"), required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026111901)
    parser.add_argument("--checkpoint", type=Path, help="Required for ML dynamics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DATA / "benchmark")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, help="Per-invocation attempt limit for bounded tests")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    add_run_arguments(runner)
    runner.set_defaults(function=run)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--input-dir", type=Path, default=DATA / "benchmark")
    comparison.add_argument("--pairs", type=int, default=8)
    comparison.add_argument("--output-prefix", type=Path, default=DATA / "radial")
    comparison.add_argument("--sweeps", default="0,1,2,4,8,16,32,64,128,256,512,750,1000")
    comparison.add_argument("--target-sweeps", default="1000", help="Required full target, or auto")
    comparison.set_defaults(function=compare)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
