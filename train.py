"""Train the fixed 10-layer, width-64 baseline; never access test labels."""
import argparse
import math
import os
from pathlib import Path
import random

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset
from common import (ROOT, DATA, sha256, read_split, load_model,
                    validate_checkpoint_data, atomic_torch_save, lock,
                    require_slurm, training_allowed)
from models import IsingGNN


@torch.inference_mode()
def validate(model, loader, scale, temperature, device):
    model.eval()
    squared = absolute = probability = count = 0
    for spins, target in loader:
        target = target.to(device)
        predicted = model(spins.to(device)) * scale
        error = (predicted - target).double()
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        probability += (torch.sigmoid(-predicted.double() / temperature)
                        - torch.sigmoid(-target.double() / temperature)).abs().sum().item()
        count += target.numel()
    return dict(mse=squared / count, mae=absolute / count, probability_mae=probability / count)


def run(args):
    require_slurm()
    if not 1 <= args.epochs <= 200 or args.batch_size < 1:
        raise ValueError("Require 1..200 epochs and positive batch size")
    prefix = args.output.resolve()
    if prefix.parent != DATA.resolve() or prefix.suffix:
        raise ValueError("--output must be an extension-free prefix in this script's directory")
    best_path = prefix.with_name(prefix.name + "_best.pt")
    last_path = prefix.with_name(prefix.name + "_last.pt")
    with lock(DATA / ".workflow.lock"):
        training_allowed()
        if args.resume and not last_path.is_file():
            raise ValueError("No last checkpoint to resume")
        if not args.resume and (best_path.exists() or last_path.exists()):
            raise ValueError("Training output exists; use --resume for an unfinished run")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        seed = 2026111806
        random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 1)
        training = read_split("train", args.data)
        validation = read_split("validation", args.data)
        config = dict(epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                      seed=seed, lr=0.001, weight_decay=1e-4, patience=30,
                      scheduler_factor=0.5, scheduler_patience=10, gradient_clip=10.0)
        model = IsingGNN(num_layers=10, hidden_dim=64, architecture="isotropic",
                         residual=True, readout="final").to(args.device)
        identity = dict(config=config, model_config=model.model_config,
                        data_sha256=training["data_sha256"],
                        splits=training["all_splits"], physics=training["physics"],
                        energy_convention=training["energy_convention"],
                        output=str(prefix), source_sha256={name: sha256(ROOT / name)
                        for name in ("train.py", "models.py", "physics.py", "common.py")})
        if validation["data_sha256"] != training["data_sha256"]:
            raise ValueError("Dataset changed while loading")
        datasets = [TensorDataset(torch.as_tensor(part["spins"], dtype=torch.float32),
                                  torch.as_tensor(part["delta"], dtype=torch.float32))
                    for part in (training, validation)]
        targets = datasets[0].tensors[1]
        # Match the original float32-target, framewise float64 RMS accumulation.
        scale = math.sqrt(sum(frame.double().square().sum().item() for frame in targets)
                          / targets.numel())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Degenerate training RMS")
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                                  generator=generator)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        history, best, bad_epochs, start = [], float("inf"), 0, 0
        if args.resume:
            _, saved = load_model(last_path, "cpu")
            validate_checkpoint_data(saved, training)
            if saved["identity"] != identity or saved["target_scale"] != scale:
                raise ValueError("Resume hyperparameters, data, source, or target scale differ")
            if saved["complete"] or saved["epoch"] >= args.epochs or saved["bad_epochs"] >= 30:
                raise ValueError("Training already complete; refusing further fitting")
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            generator.set_state(saved["loader_rng"])
            if args.device == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            history, best, bad_epochs, start = (saved[key] for key in
                                               ("history", "best", "bad_epochs", "epoch"))
        for epoch in range(start, args.epochs):
            model.train()
            total = count = 0
            for spins, target in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = (model(spins.to(args.device)) - target.to(args.device) / scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; last complete epoch preserved")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.item() * len(spins)
                count += len(spins)
            metrics = validate(model, val_loader, scale, training["physics"]["temperature"], args.device)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite validation metrics")
            scheduler.step(metrics["mse"])
            improved = metrics["mse"] < best
            best, bad_epochs = (metrics["mse"], 0) if improved else (best, bad_epochs + 1)
            history.append(dict(epoch=epoch + 1, train_loss=total / count, validation=metrics))
            checkpoint = dict(identity=identity, model_config=model.model_config,
                model_state_dict={key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()},
                target_scale=scale, physics=training["physics"],
                energy_convention=training["energy_convention"], splits=training["all_splits"],
                data_sha256=training["data_sha256"],
                dataset_manifest_sha256=training["source_manifest_sha256"],
                epoch=epoch + 1, best=best, bad_epochs=bad_epochs, history=history,
                complete=epoch + 1 == args.epochs or bad_epochs >= 30,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                loader_rng=generator.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else [])
            if improved:
                atomic_torch_save(best_path, checkpoint)
            atomic_torch_save(last_path, checkpoint)
            print(f"epoch={epoch + 1} loss={total / count:.8g} validation={metrics}", flush=True)
            if checkpoint["complete"]:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Completed production dataset directory or compact NPZ")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DATA / "trained",
                         help="Prefix producing PREFIX_best.pt and PREFIX_last.pt")
    parser.add_argument("--resume", action="store_true", help="Resume this prefix's unfinished last checkpoint")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
