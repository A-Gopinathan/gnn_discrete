"""Single-chain large-lattice GNN demonstration; never launches an ED track."""

import argparse

from common import DATA
from dynamical_benchmark import add_run_arguments, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, track=False)
    parser.set_defaults(track="ml", length=50, sweeps=1000, trial=0,
                        seed=2026111902, output_dir=DATA / "large_scale")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
