"""All-flip parity on validation; --test permanently closes this bundle to training."""
import argparse
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (DATA, sha256, read_split, load_model,
                    validate_checkpoint_data, require_slurm, lock, expose_test)


def run(args):
    require_slurm()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    split = "test" if args.test else "validation"
    output = (args.output or DATA / f"parity_{split}").resolve()
    if output.parent != DATA.resolve() or output.suffix:
        raise ValueError("--output must be an extension-free prefix in this script's directory")
    with lock(DATA / ".workflow.lock"):
        if any(output.with_suffix(suffix).exists() for suffix in (".png", ".pdf", ".npz")):
            raise ValueError("Output exists; choose a fresh --output prefix")
        torch.set_num_threads(1)
        model, checkpoint = load_model(args.checkpoint, args.device)
        checkpoint_sha = sha256(args.checkpoint)
        if args.test:
            expose_test()  # Before even loading test labels, under the shared training lock.
        data = read_split(split, args.data)
        validate_checkpoint_data(checkpoint, data)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(data["spins"]), args.batch_size):
                spins = torch.as_tensor(data["spins"][start:start + args.batch_size], device=args.device)
                predictions.append((model(spins) * checkpoint["target_scale"]).cpu().numpy())
        predicted = np.concatenate(predictions)
        exact = data["delta"].astype(np.float64)
        error = predicted.astype(np.float64) - exact
        if not np.isfinite(error).all():
            raise FloatingPointError("Nonfinite predictions")
        mse = float(np.mean(error ** 2))
        metrics = dict(mae=float(np.mean(np.abs(error))), mse=mse, rmse=float(np.sqrt(mse)))
        temperature = data["physics"]["temperature"]
        probability_mae = float(np.mean(np.abs(
            np.exp(-np.logaddexp(0, predicted.astype(np.float64) / temperature))
            - np.exp(-np.logaddexp(0, exact / temperature)))))
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
                             "mathtext.fontset": "cm", "font.size": 11})
        fig, ax = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
        ax.scatter(exact.ravel(), predicted.ravel(), s=2, alpha=0.25, linewidths=0, rasterized=True)
        low, high = min(exact.min(), predicted.min()), max(exact.max(), predicted.max())
        pad = max(float(high - low) * 0.04, 1e-6)
        limits = (low - pad, high + pad)
        ax.plot(limits, limits, color="black", linewidth=0.8)
        ax.set(xlim=limits, ylim=limits, xlabel=r"ED $\Delta E$", ylabel=r"GNN $\Delta E$")
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.96, "\n".join(f"{key.upper()} = {value:.4g}" for key, value in metrics.items()),
                transform=ax.transAxes, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        for suffix in (".png", ".pdf"):
            fig.savefig(output.with_suffix(suffix), dpi=300)
        plt.close(fig)
        temporary = output.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, exact=exact, predicted=predicted, trial=data["trial"],
                sweep=data["sweep"], split=split, split_ids=data["split_ids"],
                data_sha256=data["data_sha256"], checkpoint_sha256=checkpoint_sha,
                target_scale=checkpoint["target_scale"], probability_mae=probability_mae, **metrics)
        os.replace(temporary, output.with_suffix(".npz"))
        print(f"{split}: {metrics}, probability_mae={probability_mae:.8g}; {output}.{{png,pdf,npz}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test", action="store_true", help="Expose held-out test and permanently disable training")
    parser.add_argument("--output", type=Path, help="Fresh prefix; default parity_validation or parity_test")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
