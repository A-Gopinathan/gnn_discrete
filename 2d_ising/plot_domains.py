"""Plot completed GNN domain snapshots, with +1 red and -1 blue."""

import argparse
from pathlib import Path

from common import DATA
from dynamical_benchmark import read_trajectory, sweep_list


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "large_scale" / "ml_000.npz")
    parser.add_argument("--sweeps", default="0,10,50,100,250,500,1000")
    parser.add_argument("--output-prefix", type=Path, default=DATA / "domains")
    args = parser.parse_args(argv)
    data = read_trajectory(args.input, complete=True)
    if data["settings"]["track"] != "ml":
        raise ValueError("Domain demonstration requires an ML trajectory")
    sweeps = sweep_list(args.sweeps)
    if max(sweeps) > int(data["target_sweeps"]):
        raise ValueError("Requested snapshots are missing")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
    columns = min(4, len(sweeps))
    rows = (len(sweeps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.65 * columns, 2.7 * rows),
                             squeeze=False, layout="constrained")
    for axis, sweep in zip(axes.flat, sweeps):
        # Storage is [x,y]; transpose so horizontal plot coordinates are x.
        axis.imshow(data["spins"][sweep].T, origin="lower", interpolation="nearest",
                    cmap=ListedColormap(["tab:blue", "tab:red"]), vmin=-1, vmax=1)
        axis.set(title=f"Sweep {sweep}", xlabel=r"$x$", ylabel=r"$y$")
    for axis in list(axes.flat)[len(sweeps):]:
        axis.set_visible(False)
    fig.suptitle(r"$s=+1$: red   $s=-1$: blue", fontsize=11)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(str(args.output_prefix) + "." + suffix, dpi=220)
    plt.close(fig)
    print(f"Saved {args.output_prefix}.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
