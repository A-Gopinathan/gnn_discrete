"""Run the frozen GNN in batched independent chains, not large-lattice training."""
import argparse
from pathlib import Path

from campaign_tools import verify
from prepare_campaign import prepare, sha256, verify_current_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, help="required for a new campaign")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, help="new campaign default: 50")
    parser.add_argument("--sweeps", type=int, help="new campaign default: 1000")
    parser.add_argument("--chains", type=int, help="new campaign default: 3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--stop-after-attempts", type=int, default=0)
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.resume:
        metadata = verify(args.output)
        verify_current_sources(metadata)
        for key in ("length", "sweeps", "chains"):
            if getattr(args, key) is not None and getattr(args, key) != metadata[key]:
                parser.error("--resume must use the exact frozen " + key)
        if args.checkpoint is not None and sha256(args.checkpoint) != metadata["model_sha256"]:
            parser.error("--resume checkpoint differs from the frozen model")
    else:
        if args.output.exists():
            parser.error("Output already exists; use --resume to verify and continue")
        if args.checkpoint is None:
            parser.error("A new campaign requires --checkpoint")
        metadata = prepare(args.output, args.checkpoint,
                           50 if args.length is None else args.length,
                           1000 if args.sweeps is None else args.sweeps,
                           3 if args.chains is None else args.chains)
    rounds = metadata["length"] ** 3 * metadata["sweeps"]
    print(f"L={metadata['length']}, {metadata['chains']} independent chains batched together, "
          f"{metadata['sweeps']} sweeps: {rounds:,} rounds ({rounds * metadata['chains']:,} total attempts).",
          flush=True)
    print("Runtime must be measured on the target GPU; this may require multiple allocations. "
          "Resume uses checkpoints; the L12 ordering time is not an L50 prediction.", flush=True)
    from dynamical_benchmark import run
    args.track, args.trial = "ml", None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
