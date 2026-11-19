"""Portable paired ED/GNN campaign preparation, execution and directional plots."""
import argparse
import csv
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import warnings

import numpy as np

from campaign_tools import verify
from prepare_campaign import ROOT, prepare, sha256, verify_current_sources


def run(args):
    """Run the original engines; exit 75 means checkpointed but unfinished."""
    campaign = args.output.expanduser().resolve()
    metadata = verify(campaign)
    verify_current_sources(metadata)
    trials = list(range(metadata["chains"])) if args.trial is None else args.trial
    if (not trials or len(set(trials)) != len(trials)
            or any(t < 0 or t >= metadata["chains"] for t in trials)):
        raise ValueError("Trials must be unique campaign IDs")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Simulation requires a Slurm allocation; preparation/plotting do not")
    if args.track == "ml":
        from simulation_ml import run as simulate
        options = argparse.Namespace(campaign=campaign, output=campaign / "ml", trials=trials,
                                     device=args.device, max_seconds=args.max_seconds,
                                     stop_after_attempts=args.stop_after_attempts,
                                     checkpoint_seconds=args.checkpoint_seconds, fast=False)
        result = simulate(options)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["status"] == "complete" else 75
    executable = campaign / "source/potts_dynamics_ed"
    if not executable.is_file():
        executable = ROOT / "potts_dynamics_ed"
    if not executable.is_file():
        raise FileNotFoundError("Build the original Armadillo engine with make before ED execution")
    for trial in trials:
        command = [str(executable), "--campaign", str(campaign), "--output", str(campaign / "ed"),
                   "--trial", str(trial), "--max-seconds", str(args.max_seconds),
                   "--stop-after-attempts", str(args.stop_after_attempts),
                   "--checkpoint-seconds", str(args.checkpoint_seconds)]
        child = subprocess.Popen(command)
        handlers = {}
        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)
        try:
            for signum in (signal.SIGUSR1, signal.SIGTERM):
                previous = signal.getsignal(signum)
                if previous is not None:
                    handlers[signum] = previous
                    signal.signal(signum, forward)
            code = child.wait()
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)
        if code:
            return code if code > 0 else 128 - code
        marker = campaign / "ed" / f"trial_{trial:03d}" / "COMPLETE"
        if not marker.is_file():
            print(f"ED trial {trial}: paused; checkpoint saved, campaign unfinished", flush=True)
            return 75
        if metadata["campaign_id"] not in marker.read_text().splitlines():
            raise ValueError("ED completion marker has a different campaign identity")
    return 0


def correlations(states):
    """C_axis(r) = <1.5 delta(state_i,state_i+r) - 0.5>, with periodic axes."""
    if (states.ndim != 3 or len(set(states.shape)) != 1
            or np.any(states > 2) or np.any(states < 0)):
        raise ValueError("Expected a cubic array of orbital labels 0,1,2")
    return np.array([[1.5 * np.mean(states == np.roll(states, -r, axis=axis)) - 0.5
                      for r in range(states.shape[0] // 2 + 1)] for axis in range(3)])


def committed_sweep(directory, campaign, trial, track):
    """Atomic progress.csv is published only after checkpoint/raw commitment."""
    progress = directory / "progress.csv"
    if not progress.exists():
        return -1
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("campaign_id") != campaign["campaign_id"] or metadata.get("trial") != trial:
        raise ValueError("Trial/campaign identity mismatch: " + str(directory))
    if track == "ml" and metadata.get("model_sha256") != campaign["model_sha256"]:
        raise ValueError("ML model identity mismatch: " + str(directory))
    with progress.open() as handle:
        rows = list(csv.DictReader(handle))
    n = campaign["length"] ** 3
    if not rows or any(int(row["sweep"]) != i or int(row["attempts"]) != i * n
                       for i, row in enumerate(rows)):
        raise ValueError("Invalid committed sweep sequence: " + str(progress))
    last = len(rows) - 1
    if last > campaign["sweeps"] or (directory / "states.bin").stat().st_size < (last + 1) * n:
        raise ValueError("Raw trajectory shorter than published progress: " + str(directory))
    with (directory / "states.bin").open("rb") as handle:
        initial = handle.read(n)
    if hashlib.sha256(initial).hexdigest() != campaign["initial_sha256"][f"trial_{trial:03d}.bin"]:
        raise ValueError("Initial frame differs from paired campaign input: " + str(directory))
    return last


def mean_sem(values):
    if len(values) == 0:
        raise ValueError("Cannot average an empty ensemble")
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.full_like(mean, np.nan)
    return mean, sem


def plot(args):
    """Use committed paired frames, or exactly the cohorts in a curated CSV."""
    sweeps = sorted(set(args.sweeps))
    if not sweeps or min(sweeps) < 0:
        raise ValueError("Requested sweeps must be nonnegative")
    output = args.output.expanduser().resolve()
    values, selected = {}, {}
    details = dict(sweeps=sweeps, uncertainty="sample SEM across independent chains; paired ML-minus-ED differences",
                   observable="C_axis(r)=mean_sites[1.5*(state_i==state_(i+r))-0.5], periodic, unconnected",
                   source_sha256=sha256(Path(__file__)))
    if args.table is not None:
        if args.available or args.campaign is not None or args.ed_dir is not None or args.ml_dir is not None:
            raise ValueError("--table preserves its own cohorts; do not combine with live-data options")
        table = args.table.expanduser().resolve()
        records = {}
        with table.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["sweep", "trial", "track", "axis", "r", "correlation"]:
                raise ValueError("Expected report per_chain_correlations.csv columns")
            for row in reader:
                sweep, trial, radius = (int(row[k]) for k in ("sweep", "trial", "r"))
                track, axis, value = row["track"], row["axis"], float(row["correlation"])
                if (track not in ("ed", "ml") or axis not in "xyz" or len(axis) != 1
                        or min(sweep, trial, radius) < 0 or not np.isfinite(value) or not -.5 <= value <= 1):
                    raise ValueError("Invalid correlation table row")
                key = (sweep, trial, track, axis, radius)
                if key in records:
                    raise ValueError("Duplicate correlation table row")
                records[key] = value
        radii = sorted({key[4] for key in records})
        if len(radii) < 2 or radii != list(range(len(radii))):
            raise ValueError("Expected consecutive radii starting at zero")
        length = 2 * radii[-1]
        for sweep in sweeps:
            cohorts = {track: {key[1] for key in records if key[0] == sweep and key[2] == track}
                       for track in ("ed", "ml")}
            if not cohorts["ed"] or cohorts["ed"] != cohorts["ml"]:
                raise ValueError(f"Missing or unpaired table cohort at sweep {sweep}")
            ids = selected[sweep] = sorted(cohorts["ed"])
            values[sweep] = {}
            for track in ("ed", "ml"):
                try:
                    curves = np.array([[[records[(sweep, trial, track, axis, r)] for r in radii]
                                        for axis in "xyz"] for trial in ids])
                except KeyError as exc:
                    raise ValueError("Incomplete directional table curve: " + str(exc)) from exc
                if not np.all(curves[:, :, 0] == 1):
                    raise ValueError("Expected C(0)=1")
                values[sweep][track] = curves
        details.update(table=str(table), table_sha256=sha256(table), cohort="curated table (unchanged per-sweep IDs)")
    else:
        campaign_dir = args.campaign.expanduser().resolve() if args.campaign else output
        campaign = verify(campaign_dir)
        if max(sweeps) > campaign["sweeps"]:
            raise ValueError("Requested sweeps exceed the campaign")
        roots = {track: (getattr(args, track + "_dir").expanduser().resolve()
                         if getattr(args, track + "_dir") else campaign_dir / track)
                 for track in ("ed", "ml")}
        if args.campaign is None and args.plot_dir is None:
            output = campaign_dir / "plots"
        availability = {trial: {track: committed_sweep(root / f"trial_{trial:03d}", campaign, trial, track)
                                for track, root in roots.items()} for trial in range(campaign["chains"])}
        length = campaign["length"]
        hashes = {}
        for sweep in sweeps:
            required = sweep if args.available else max(sweeps)
            ids = selected[sweep] = [trial for trial, status in sorted(availability.items())
                                    if status["ed"] >= required and status["ml"] >= required]
            if not ids:
                raise ValueError(f"No matched chains at sweep {sweep}; request earlier sweeps or --available")
            values[sweep] = {}
            for track in ("ed", "ml"):
                curves = []
                for trial in ids:
                    with (roots[track] / f"trial_{trial:03d}" / "states.bin").open("rb") as handle:
                        handle.seek(sweep * length ** 3)
                        content = handle.read(length ** 3)
                    if len(content) != length ** 3:
                        raise ValueError("Incomplete requested frame")
                    curves.append(correlations(np.frombuffer(content, dtype=np.uint8).reshape((length,) * 3)))
                    hashes[f"{track}/trial_{trial:03d}/sweep_{sweep}"] = hashlib.sha256(content).hexdigest()
                values[sweep][track] = np.stack(curves)
        details.update(campaign=str(campaign_dir), campaign_id=campaign["campaign_id"],
                       availability=availability, frame_sha256=hashes,
                       cohort="available per-sweep pairs" if args.available else "fixed through maximum requested sweep")
        if args.available:
            warnings.warn("Available cohorts can change with sweep and may favor faster ED chains; IDs and n are exported")
    if args.plot_dir is not None:
        output = args.plot_dir.expanduser().resolve()
    if args.table is None:
        if output == campaign_dir or any(output == root or root in output.parents for root in roots.values()):
            raise ValueError("Plot output must not overwrite campaign or trajectory inputs")
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                         "mathtext.fontset": "cm"})
    radii = np.arange(length // 2 + 1)
    archive = dict(sweeps=np.array(sweeps), radii=radii, axes=np.array(list("xyz")))
    summary, individuals, clipped = [], [], []
    statistics = {}
    for sweep in sweeps:
        ids = selected[sweep]
        archive[f"sweep_{sweep}_ids"] = np.array(ids)
        statistics[sweep] = {}
        for track, curves in [*values[sweep].items(), ("ml_minus_ed", values[sweep]["ml"] - values[sweep]["ed"])]:
            archive[f"sweep_{sweep}_{track}"] = curves
            mean, sem = statistics[sweep][track] = mean_sem(curves)
            archive[f"sweep_{sweep}_{track}_mean"] = mean
            archive[f"sweep_{sweep}_{track}_sem"] = sem
            for a, axis in enumerate("xyz"):
                summary.extend([sweep, track, axis, int(r), len(ids), mean[a, r], sem[a, r]] for r in radii)
                if track in ("ed", "ml"):
                    if np.any(mean[a] < 0) or np.any(mean[a] - sem[a] < 0):
                        clipped.append(dict(sweep=sweep, axis=axis, track=track))
                    for i, trial in enumerate(ids):
                        individuals.extend([sweep, trial, track, axis, int(r), curves[i, a, r]] for r in radii)
    groups = [("dynamical_early", [s for s in sweeps if s <= 200]),
              ("dynamical_late", [s for s in sweeps if s > 200])]
    for filename, panel_sweeps in groups:
        if not panel_sweeps:
            continue
        fig, axes = plt.subplots(len(panel_sweeps), 3, figsize=(12, 3.3 * len(panel_sweeps)),
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        for row_index, sweep in enumerate(panel_sweeps):
            for a, (ax, axis) in enumerate(zip(axes[row_index], "xyz")):
                for track, label, color, marker, style in [("ed", "ED", "#2166ac", "o", "-"),
                                                          ("ml", "GNN", "#c62828", "s", "none")]:
                    mean, sem = statistics[sweep][track]
                    ax.errorbar(radii, mean[a], yerr=sem[a] if len(selected[sweep]) > 1 else None,
                                color=color, marker=marker, linestyle=style, markersize=4, linewidth=1.4,
                                markerfacecolor="none" if track == "ml" else color, capsize=2,
                                label=f"{label} (n={len(selected[sweep])})")
                ax.set(title=f"Sweep {sweep}, {axis}", xlim=(-.1, radii[-1] + .1), xticks=radii)
                if not args.full_range:
                    ax.set_ylim(0, 1.05)
                ax.grid(alpha=.18)
                ax.legend(loc="upper right", fontsize=8)
                if row_index == len(panel_sweeps) - 1:
                    ax.set_xlabel("Separation r")
                if a == 0:
                    ax.set_ylabel("Correlation")
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{extension}", dpi=200)
        plt.close(fig)
    np.savez_compressed(output / "correlations.npz", **archive)
    for filename, header, rows in [
        ("correlation_summary.csv", ["sweep", "track", "axis", "r", "chains", "mean", "sem"], summary),
        ("per_chain_correlations.csv", ["sweep", "trial", "track", "axis", "r", "correlation"], individuals)]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    details.update(length=length, trial_ids=selected, y_limits="automatic" if args.full_range else [0, 1.05],
                   error_bars_below_zero=clipped)
    (output / "provenance.json").write_text(json.dumps(details, indent=2) + "\n")
    if clipped and not args.full_range:
        warnings.warn("y>=0 hides negative means/error-bar portions; full values saved in NPZ/CSV. Use --full-range.")
    print("Matched pairs: " + str({s: len(ids) for s, ids in selected.items()}))
    print("Figures and full numerical data: " + str(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="freeze sources/model and paired initial states only")
    prep.add_argument("--checkpoint", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--length", type=int, default=12)
    prep.add_argument("--sweeps", type=int, default=1000)
    prep.add_argument("--chains", type=int, default=30)
    prep.add_argument("--backup-root", type=Path)
    execute = sub.add_parser("run", help="run inside an existing Slurm allocation; no submission")
    execute.add_argument("--output", type=Path, required=True, help="prepared campaign/result root")
    execute.add_argument("--track", choices=("ed", "ml"), required=True)
    execute.add_argument("--trial", "--trials", type=int, nargs="+", help="default: all campaign trials")
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--max-seconds", type=float, default=0)
    execute.add_argument("--stop-after-attempts", type=int, default=0)
    execute.add_argument("--checkpoint-seconds", type=float, default=60)
    plotting = sub.add_parser("plot", help="directional correlations from committed frames or a curated table")
    plotting.add_argument("--output", type=Path, required=True,
                          help="new RUN root (plots under RUN/plots), or figure directory with --campaign/--table")
    plotting.add_argument("--campaign", type=Path, help="explicit campaign, including preserved old campaigns")
    plotting.add_argument("--ed-dir", type=Path)
    plotting.add_argument("--ml-dir", type=Path)
    plotting.add_argument("--table", type=Path, help="curated report per_chain_correlations.csv; retains exact cohorts")
    plotting.add_argument("--plot-dir", type=Path, help="explicit new figure-output directory")
    plotting.add_argument("--sweeps", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    plotting.add_argument("--available", action="store_true", help="live data: per-sweep matched cohort instead of fixed")
    plotting.add_argument("--full-range", action="store_true", help="show negative correlations/error bars")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        options = vars(args).copy()
        del options["command"]
        metadata = prepare(**options)
        print("Prepared campaign " + metadata["campaign_id"] + "; no simulations submitted")
        return 0
    if args.command == "run":
        return run(args)
    plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
