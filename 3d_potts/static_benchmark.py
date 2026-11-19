"""Replot saved Potts test predictions, or evaluate a trusted checkpoint in Slurm."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np


STAGES = (("0_50", 0, 50), ("51_200", 51, 200),
          ("201_400", 201, 400), ("401_800", 401, 800))
PHYSICS = dict(temperature=0.1, filling=0.5, alpha=1.0, beta=-0.5, gamma=-0.7,
               energy="finite-temperature fermionic internal energy",
               target="E(single-site target configuration) - E(original configuration)",
               classical_energy=False, per_site_normalization=False, boundary="periodic",
               orbitals=["X", "Y", "Z"], index_order="x*L*L+y*L+z", edge_axes=["x", "y", "z"])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_data(root, config):
    """Verify the original prepared format and select whole held-out chains.

    Checksums establish artifact identity, not independent ED recomputation.
    Historical paths in the fingerprinted metadata are intentionally unchanged.
    """
    root = Path(root).resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    payload = dict(metadata)
    fingerprint = payload.pop("fingerprint")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if actual != fingerprint or fingerprint != config["fingerprint"]:
        raise ValueError("Dataset/checkpoint fingerprint mismatch")
    if metadata != config["metadata"] or metadata["physics"] != PHYSICS:
        raise ValueError("Dataset/checkpoint metadata or physics mismatch")
    expected = {"states.npy", "targets.npy", "trials.npy", "sweeps.npy", "inventory.json", "splits.json"}
    if metadata["format_version"] != 1 or set(metadata["files_sha256"]) != expected:
        raise ValueError("Unsupported prepared dataset format/inventory")
    for name, digest in metadata["files_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError("Dataset checksum mismatch: " + name)
    splits = json.loads((root / "splits.json").read_text())
    if splits != config["splits"] or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Checkpoint/split mismatch")
    groups = list(splits.values())
    flat = [trial for group in groups for trial in group]
    if any(not group for group in groups) or any(type(t) is not int for t in flat) or len(set(flat)) != len(flat):
        raise ValueError("Splits must be nonempty, disjoint integer chain IDs")
    arrays = {key: np.load(root / (key + ".npy"), mmap_mode="r", allow_pickle=False)
              for key in ("states", "targets", "trials", "sweeps")}
    graphs, sites = metadata["graphs"], metadata["length"] ** 3
    for key, shape, dtype in (("states", (graphs, sites), np.uint8),
                              ("targets", (graphs, sites, 3), np.float32),
                              ("trials", (graphs,), np.int64), ("sweeps", (graphs,), np.int64)):
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError("Prepared array shape/dtype mismatch: " + key)
    if set(flat) != set(arrays["trials"].tolist()):
        raise ValueError("Split IDs do not cover the dataset")
    selected = np.isin(arrays["trials"], splits["test"])
    return {key: value[selected] for key, value in arrays.items()}, metadata


def validate_arrays(arrays):
    states, targets, predictions = (arrays[k] for k in ("states", "targets", "predictions"))
    if states.ndim != 2 or not states.size or not np.issubdtype(states.dtype, np.integer):
        raise ValueError("Expected integer states[graph,site]")
    if targets.shape != states.shape + (3,) or predictions.shape != targets.shape:
        raise ValueError("Expected targets/predictions[graph,site,3]")
    if np.any((states < 0) | (states > 2)) or not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Invalid states or nonfinite energy changes")
    for key in ("trials", "sweeps"):
        if arrays[key].shape != (len(states),) or not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError("Expected integer " + key + "[graph]")
    if np.any(arrays["trials"] < 0) or np.any(arrays["sweeps"] < 0):
        raise ValueError("Negative trial/sweep IDs")
    if len(np.unique(np.stack((arrays["trials"], arrays["sweeps"]), axis=1), axis=0)) != len(states):
        raise ValueError("Duplicate trial/sweep snapshots")
    for values in (targets, predictions):
        if np.any(np.take_along_axis(values, states[..., None], axis=2) != 0):
            raise ValueError("Self transitions must be exactly zero")


def nonself_values(arrays, selected):
    # Membership, never label magnitude, decides which transitions are excluded.
    mask = np.arange(3)[None, None, :] != arrays["states"][selected, :, None]
    return tuple(arrays[key][selected][mask].astype(np.float64)
                 for key in ("targets", "predictions"))


def metrics(truth, prediction):
    if not len(truth):
        raise ValueError("Empty parity selection")
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    return dict(transition_count=len(error), mae=float(np.mean(np.abs(error))),
                mse=mse, rmse=float(np.sqrt(mse)))


def infer(checkpoint_path, dataset_path, device):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Fresh inference requires a Slurm allocation; --predictions only replots")
    import torch
    from local_gnn import full_graph
    from potts_gnn import PottsGNN

    checkpoint_path = Path(checkpoint_path).resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    with (checkpoint_path.parent / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Original checkpoints contain optimizer/RNG metadata. Only load trusted files.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config["model"] != {"num_layers": 3, "hidden_dim": 64}:
            raise ValueError("Expected production three-layer, width-64 model")
        if config["settings"].get("smoke") or config["settings"].get("overfit") or config["selection_metric"] != "validation_mse":
            raise ValueError("Expected a validation-selected production checkpoint")
        scale = float(checkpoint["target_scale"])
        if not np.isfinite(scale) or scale <= 0 or scale != config["target_scale"]:
            raise ValueError("Invalid or inconsistent training target scale")
        arrays, metadata = load_test_data(dataset_path, config)
        model = PottsGNN(**config["model"]).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters")
        # Same sibling marker and lock as the original trainer. Never clear it.
        exposure = checkpoint_path.parent / "test_exposure.json"
        if not exposure.exists():
            temporary = exposure.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(dict(checkpoint=str(checkpoint_path),
                fingerprint=metadata["fingerprint"],
                note="Held-out test access began; further fitting of this run is forbidden."), indent=2) + "\n")
            temporary.replace(exposure)
        predictions = []
        with torch.inference_mode():
            graph = full_graph(torch.as_tensor(arrays["states"][0], dtype=torch.long, device=device), metadata["length"])
            for index, state in enumerate(arrays["states"]):
                graph.x = torch.as_tensor(state, dtype=torch.long, device=device)[:, None]
                # Match the original evaluator: promote BEFORE restoring the RMS.
                predictions.append((model(graph).double() * scale).cpu().numpy())
                if (index + 1) % 100 == 0:
                    print(f"Predicted {index + 1}/{len(arrays['states'])} test graphs", flush=True)
        arrays["predictions"] = np.stack(predictions)
    provenance = dict(checkpoint_sha256=sha256(checkpoint_path), checkpoint_epoch=checkpoint["epoch"],
                      fingerprint=metadata["fingerprint"], target_scale=scale,
                      physics=PHYSICS, splits=config["splits"], split="test",
                      model_source_sha256=sha256(Path(__file__).with_name("potts_gnn.py")))
    return arrays, provenance


def plot(arrays, prefix, stages=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(Path(matplotlib.get_data_path()) / "fonts/ttf/cmr10.ttf"))
    groups = {"overall": np.ones(len(arrays["sweeps"]), dtype=bool)}
    if stages:
        groups.update({key: (arrays["sweeps"] >= low) & (arrays["sweeps"] <= high)
                       for key, low, high in STAGES})
        if not np.all(np.sum(list(groups.values())[1:], axis=0) == 1):
            raise ValueError("Requested stages must cover every snapshot exactly once")
    values = {key: nonself_values(arrays, selected) for key, selected in groups.items()}
    statistics = {key: metrics(*pair) for key, pair in values.items()}
    truth, prediction = values["overall"]
    low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
    pad = max(float(high - low) * 0.04, 1e-6)
    bounds = (low - pad, high + pad)

    def panel(ax, key, color, title=None):
        ax.scatter(*values[key], s=1, alpha=.035, color=color, linewidths=0, rasterized=True)
        ax.plot(bounds, bounds, color=".2", ls="--", lw=.8)
        ax.set(xlim=bounds, ylim=bounds, xlabel=r"ED $\Delta E$", ylabel=r"GNN $\Delta E$")
        ax.set_aspect("equal", adjustable="box")
        if title:
            ax.set_title(title)
        ax.text(.04, .96, "\n".join(f"{metric.upper()} = {statistics[key][metric]:.5g}"
                for metric in ("mae", "mse", "rmse")),
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.95))
        ax.grid(alpha=.15)

    with plt.rc_context({"font.family": "cmr10", "mathtext.fontset": "cm", "font.size": 11,
                         "axes.formatter.use_mathtext": True, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        figure, ax = plt.subplots(figsize=(4.8, 4.8), layout="constrained")
        panel(ax, "overall", "#21658a")
        for suffix in (".png", ".pdf"):
            figure.savefig(str(prefix) + suffix, dpi=250)
        plt.close(figure)
        if stages:
            figure, axes = plt.subplots(2, 2, figsize=(9, 9), layout="constrained")
            for ax, (key, first, last), color in zip(axes.flat, STAGES, ("#b55335", "#247a78", "#535895", "#32774d")):
                panel(ax, key, color, f"Sweeps {first}-{last}")
            for suffix in (".png", ".pdf"):
                figure.savefig(str(prefix) + "_stages" + suffix, dpi=250)
            plt.close(figure)
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="Saved test_predictions.npz; no model inference")
    source.add_argument("--checkpoint", type=Path, help="Trusted local production checkpoint")
    parser.add_argument("--dataset", type=Path, help="Original prepared dataset; required with --checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-prefix", type=Path, default=Path("parity"))
    parser.add_argument("--stages", action="store_true", help="Also plot sweeps 0-50, 51-200, 201-400, 401-800")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.dataset):
        parser.error("--dataset must be supplied exactly when --checkpoint is supplied")
    if args.checkpoint and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Fresh inference requires a Slurm allocation")
    prefix = args.output_prefix.resolve()
    suffixes = [".png", ".pdf", "_metrics.npz"]
    if args.stages:
        suffixes += ["_stages.png", "_stages.pdf"]
    if args.checkpoint:
        suffixes += ["_predictions.npz"]
    for suffix in suffixes:
        path = Path(str(prefix) + suffix)
        if path.exists():
            raise FileExistsError("Choose a fresh output prefix: " + str(path))
    if args.dataset and (prefix.parent == args.dataset.resolve() or args.dataset.resolve() in prefix.parents):
        raise ValueError("Do not write benchmark outputs into the prepared dataset")
    if args.predictions:
        with np.load(args.predictions, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("predictions", "targets", "states", "trials", "sweeps")}
        provenance = dict(predictions_sha256=sha256(args.predictions), predictions=str(args.predictions.resolve()))
    else:
        arrays, provenance = infer(args.checkpoint, args.dataset, args.device)
    validate_arrays(arrays)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        np.savez_compressed(str(prefix) + "_predictions.npz", **arrays,
                            provenance=json.dumps(provenance, sort_keys=True))
    statistics = plot(arrays, prefix, args.stages)
    provenance.update(graph_count=len(arrays["states"]), test_chain_ids=np.unique(arrays["trials"]).tolist(),
                      all_nonself_points_plotted=True, units="total fermionic internal-energy difference; no N division")
    np.savez_compressed(str(prefix) + "_metrics.npz", groups=np.array(list(statistics)),
                        **{key: np.array([s[key] for s in statistics.values()])
                           for key in ("transition_count", "mae", "mse", "rmse")},
                        provenance=json.dumps(provenance, sort_keys=True))
    print(json.dumps(dict(output_prefix=str(prefix), statistics=statistics, **provenance), indent=2))


if __name__ == "__main__":
    main()
