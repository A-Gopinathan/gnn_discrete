"""Plot actual committed Potts snapshots, without Wolfram or a TeX installation.

Outputs: PREFIX_slice_triptych.{png,pdf}, PREFIX_late_inset.{png,pdf},
PREFIX_lattice.npz. The NPZ stores the plotted states, coordinates, and provenance.
X/Y/Z = 0/1/2; flattened site order is x*L**2+y*L+z (C order).

The inset is an unwrapped periodic crop of measured orbital labels. Gray rods
indicate nearest-neighbor lattice geometry inside the crop, NOT Hamiltonian
matrix elements, orbital-dependent hopping strengths, or all periodic bonds.
Dumbbells illustrate orbital orientation, not spin vectors or density isosurfaces.
One selected chain is illustrative, not an ensemble or an equilibration claim.

Rendering requires NumPy and Matplotlib only. Reading ML checkpoints additionally
requires PyTorch; no model inference, PyG, or original-project imports are used.
Campaign verification uses the sibling campaign_tools module. Use a fresh output
prefix outside the campaign/trial; existing figures are never overwritten.
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


COLORS = np.array([[0.88, 0.06, 0.09], [0.06, 0.23, 0.92], [0.58, 0.08, 0.76]])
FNV_OFFSET = 14695981039346656037


def fnv1a(content, value=FNV_OFFSET):
    """The original ED engine's unsigned 64-bit FNV-1a checksum."""
    for byte in content:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def validate_states(states, length):
    if (states.dtype != np.uint8 or states.shape != (length,) * 3
            or np.any(states > 2)):
        raise ValueError("Expected uint8[L,L,L] orbital states 0,1,2")
    return states


def periodic_crop(states, origin, size):
    """Return states and global xyz for local C-ordered coordinates [i,j,k]."""
    length = states.shape[0]
    validate_states(states, length)
    if size not in (4, 6) or size > length or len(origin) != 3:
        raise ValueError("Inset size must be 4 or 6, no larger than L")
    xyz = (np.moveaxis(np.indices((size,) * 3), 0, -1) + np.asarray(origin)) % length
    return states[tuple(np.moveaxis(xyz, -1, 0))], xyz


def _ml_checkpoint(content, metadata, length):
    import torch

    # Original partial-sweep checkpoints contain NumPy RNG draw buffers. Allow
    # only their specific NumPy constructors, never unrestricted pickle loading.
    from numpy.core.multiarray import _reconstruct, scalar
    safe = [_reconstruct, scalar, np.ndarray, np.dtype]
    safe += [type(np.dtype(t)) for t in (np.int64, np.float64, np.uint8)]
    with torch.serialization.safe_globals(safe):
        saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)

    def plain(value):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items() if k != "checkpoint_sha256"}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    digest = hashlib.sha256(json.dumps(plain(saved), sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != saved["checkpoint_sha256"] or saved["identity"] != metadata:
        raise ValueError("ML checkpoint checksum/identity mismatch")
    n = length ** 3
    attempts = saved["attempts"]
    if type(attempts) is not int or not 0 <= attempts <= metadata["sweeps"] * n:
        raise ValueError("Invalid checkpoint attempt count")
    completed = attempts // n
    state, last = (saved[key].numpy() for key in ("state", "last_completed"))
    for array in (state, last):
        if array.shape != (n,):
            raise ValueError("Invalid checkpoint state shape")
        validate_states(array.reshape((length,) * 3), length)
    if attempts % n == 0 and not np.array_equal(state, last):
        raise ValueError("Checkpoint boundary states disagree")
    history = saved["history"]
    if len(history) != completed + 1 or any(row[:2] != [i, i * n] for i, row in enumerate(history)):
        raise ValueError("Invalid checkpoint history")
    return completed, last.tobytes(), saved["raw_sha256"], history


def _ed_checkpoint(content, metadata, campaign_dir, length):
    magic, checksum, body = content.split(b"\n", 2)
    if magic != b"POTTS_DYNAMICS_ED_CHECKPOINT_V1" or checksum.decode() != f"{fnv1a(body):016x}":
        raise ValueError("ED checkpoint checksum/header mismatch")
    rows = body.decode("ascii").splitlines()
    identity = rows[0].split()
    cfg = dict(row.split("=", 1) for row in (campaign_dir / "campaign.cfg").read_text().splitlines())
    canonical = "".join(f"{key}={cfg[key]}\n" for key in sorted(cfg)).encode()
    expected = ["identity", f"{fnv1a(canonical):016x}", metadata["campaign_id"],
                metadata["initial_hash_fnv1a64"], str(metadata["trial"])]
    if identity != expected or identity[1] != metadata["config_fingerprint_fnv1a64"]:
        raise ValueError("ED checkpoint identity mismatch")
    position = rows[1].split()
    if len(position) != 6 or position[0] != "position":
        raise ValueError("Invalid ED position")
    completed, within = map(int, position[1:3])
    if not 0 <= completed <= metadata["total_sweeps"] or not 0 <= within < length ** 3:
        raise ValueError("Invalid ED completed sweep")
    if completed == metadata["total_sweeps"] and within:
        raise ValueError("ED checkpoint beyond campaign end")
    arrays = []
    for row, name in zip(rows[3:5], ("current", "completed")):
        fields = row.split()
        if fields[0] != name or len(fields) != length ** 3 + 1 or any(v not in ("0", "1", "2") for v in fields[1:]):
            raise ValueError("Invalid ED checkpoint lattice")
        arrays.append(np.array(fields[1:], dtype=np.uint8).tobytes())
    if not within and arrays[0] != arrays[1]:
        raise ValueError("ED boundary states disagree")
    if rows[2].split()[0] != "raw_hash" or rows[6] != f"history {completed + 1}" or rows[-1] != "END":
        raise ValueError("Invalid ED checkpoint history/header")
    history = [[int(r[0]), int(r[0]) * length ** 3, int(r[1]), float(r[2])]
               for r in (row.split() for row in rows[7:-1])]
    if len(history) != completed + 1 or any(row[0] != i for i, row in enumerate(history)):
        raise ValueError("Invalid ED history sequence")
    return completed, arrays[1], int(rows[2].split()[1]), history


def read_snapshots(campaign_dir, directory, sweeps):
    """Read requested frames only after verifying the complete committed prefix.

    Matches original committed_sweep(directory,campaign,trial,track) semantics,
    with additional checkpoint and raw-prefix auditing. Never repairs/truncates
    files. Existing writer locks are opened read-only; no input files are created.
    """
    from campaign_tools import verify

    campaign_dir, directory = Path(campaign_dir).resolve(), Path(directory).resolve()
    if campaign_dir.is_file():
        campaign_dir = campaign_dir.parent
    campaign = verify(campaign_dir)
    match = re.fullmatch(r"trial_(\d+)", directory.name)
    if not match or not 0 <= int(match[1]) < campaign["chains"]:
        raise ValueError("--trial-dir must be an actual trial_NNN folder in this campaign")
    trial = int(match[1])
    length, n = campaign["length"], campaign["length"] ** 3
    with ExitStack() as stack:
        lockpath = directory.parent / (directory.name + ".lock")
        if lockpath.exists():
            import fcntl
            lock = stack.enter_context(lockpath.open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Trial has an active writer; plot a stable backup or wait for it to stop") from error
        metadata_bytes = (directory / "metadata.json").read_bytes()
        metadata = json.loads(metadata_bytes)
        track = "ml" if metadata.get("format") == "POTTS_ML_CHAIN_V1" else metadata.get("track")
        if track not in ("ed", "ml") or metadata.get("track", track) != track:
            raise ValueError("Unknown or inconsistent trial track")
        for parent in (directory.parent, directory.parent.parent):
            if parent.name in ("ed", "ml", "results_ed", "results_ml") and parent.name.removeprefix("results_") != track:
                raise ValueError("Trial directory track conflicts with metadata")
        expected = dict(campaign_id=campaign["campaign_id"], trial=trial,
                        physics={k: campaign[k] for k in ("temperature", "filling", "alpha", "beta", "gamma")})
        if track == "ml":
            expected.update(length=length, sweeps=campaign["sweeps"], model_sha256=campaign["model_sha256"],
                            initial_sha256=campaign["initial_sha256"][f"trial_{trial:03d}.bin"],
                            seed=campaign["ml_seed_base"] + trial)
        else:
            expected.update(format="POTTS_DYNAMICS_V1", linear_size=length, total_sweeps=campaign["sweeps"])
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Trial/campaign identity mismatch")
        checkpoint = directory / ("checkpoint.pt" if track == "ml" else "checkpoint.dat")
        checkpoint_bytes = checkpoint.read_bytes()
        cp_sweep, cp_last, cp_hash, history = (_ml_checkpoint(checkpoint_bytes, metadata, length)
            if track == "ml" else _ed_checkpoint(checkpoint_bytes, metadata, campaign_dir, length))
        progress_bytes = (directory / "progress.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(progress_bytes.decode())))
        if not rows or len(rows) > cp_sweep + 1:
            raise ValueError("Published progress is empty or ahead of the checkpoint")
        for i, row in enumerate(rows):
            values = [int(row["sweep"]), int(row["attempts"]), int(row["changed_sites"]), float(row["elapsed_seconds"])]
            if values[:2] != [i, i * n] or values[:3] != history[i][:3] or not np.isclose(values[3], history[i][3], rtol=1e-12, atol=1e-12):
                raise ValueError("Progress disagrees with committed checkpoint history")
        committed = len(rows) - 1
        if len(sweeps) != 3 or sweeps != sorted(set(sweeps)) or sweeps[0] < 0 or sweeps[-1] > committed:
            raise ValueError(f"Choose three increasing saved sweeps in [0,{committed}]; default is 0,100,800")
        raw_paths = [directory / name for name in ("trajectory.bin", "states.bin") if (directory / name).is_file()]
        if len(raw_paths) != 1:
            raise ValueError("Expected exactly one of trajectory.bin or legacy states.bin")
        raw = raw_paths[0]
        if raw.stat().st_size < (cp_sweep + 1) * n:
            raise ValueError("Raw trajectory is shorter than the checkpoint commitment")
        digest, fnv, frames, hashes = hashlib.sha256(), FNV_OFFSET, [], []
        with raw.open("rb") as stream:
            for sweep in range(cp_sweep + 1):
                content = stream.read(n)
                if len(content) != n:
                    raise ValueError("Short committed frame")
                states = validate_states(np.frombuffer(content, dtype=np.uint8).reshape((length,) * 3), length)
                digest.update(content)
                if track == "ed":
                    fnv = fnv1a(content, fnv)
                if sweep == 0:
                    if hashlib.sha256(content).hexdigest() != campaign["initial_sha256"][f"trial_{trial:03d}.bin"]:
                        raise ValueError("Initial frame differs from campaign")
                    if track == "ed" and f"{fnv1a(content):016x}" != metadata["initial_hash_fnv1a64"]:
                        raise ValueError("ED initial-state identity mismatch")
                if sweep in sweeps:
                    frames.append(states.copy())
                    hashes.append(hashlib.sha256(content).hexdigest())
            if content != cp_last or (digest.hexdigest() if track == "ml" else fnv) != cp_hash:
                raise ValueError("Committed trajectory checksum/last-frame mismatch")
        for path, before in ((checkpoint, checkpoint_bytes), (directory / "progress.csv", progress_bytes),
                             (directory / "metadata.json", metadata_bytes)):
            if path.read_bytes() != before:
                raise ValueError("Trial changed during validation; retry a stable snapshot")
    provenance = dict(campaign_id=campaign["campaign_id"], trial=trial, track=track, length=length,
                      committed_sweep=committed, checkpoint_sweep=cp_sweep,
                      checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                      raw_prefix_sha256=digest.hexdigest(), frame_sha256=np.array(hashes),
                      trajectory=str(raw), campaign=str(campaign_dir))
    return np.stack(frames), provenance


def _slice(ax, states, normal, index):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Ellipse

    length = states.shape[0]
    plane = [axis for axis in range(3) if axis != normal]
    labels = np.take(states, index, axis=normal)
    centers = np.indices(labels.shape).reshape(2, -1).T.astype(float)
    values = labels.ravel()
    # Projection is geometric: the orbital normal to the slice appears end-on.
    for state in range(3):
        points = centers[values == state]
        if state == normal:
            offsets, widths, heights, angle = points, .30, .30, 0
        else:
            direction = np.eye(2)[plane.index(state)]
            offsets = np.concatenate([points - .19 * direction, points + .19 * direction])
            widths, heights, angle = .34, .22, 90 * plane.index(state)
        for center in offsets:
            ax.add_patch(Ellipse(center, widths, heights, angle=angle,
                                 facecolor=COLORS[state], edgecolor="none"))
    grid = [[(i, -.5), (i, length - .5)] for i in range(length)]
    grid += [[(-.5, i), (length - .5, i)] for i in range(length)]
    ax.add_collection(LineCollection(grid, colors="0.91", linewidths=.35, zorder=-1))
    ticks = sorted(set([0, length // 2, length - 1]))
    ax.set(xlim=(-.6, length - .4), ylim=(-.6, length - .4), aspect="equal", xticks=ticks, yticks=ticks)
    ax.set_xlabel(f"${'xyz'[plane[0]]}$", color=COLORS[plane[0]], labelpad=1)
    ax.set_ylabel(f"${'xyz'[plane[1]]}$", color=COLORS[plane[1]], labelpad=0)
    ax.tick_params(length=2, labelsize=9, colors="0.35")
    for spine in ax.spines.values():
        spine.set_color("0.8")
    return labels, plane


def _mark_roi(ax, origin, size, plane, length):
    from matplotlib.patches import Rectangle

    intervals = []
    for axis in plane:
        start = int(origin[axis]) % length
        intervals.append([(start, min(size, length - start))] +
                         ([(0, size - length + start)] if start + size > length else []))
    centers = []
    for x, width in intervals[0]:
        for y, height in intervals[1]:
            ax.add_patch(Rectangle((x - .5, y - .5), width, height, fill=False,
                                  edgecolor="0.12", linewidth=1.25, linestyle=(0, (3, 2)), zorder=5))
            centers.append((x + (width - 1) / 2, y + (height - 1) / 2))
    return centers[0]


def _inset(ax, states):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    size = states.shape[0]
    xyz = np.indices(states.shape).reshape(3, -1).T.astype(float)
    bonds = [(point, point + np.eye(3)[axis]) for point in xyz for axis in range(3) if point[axis] < size - 1]
    ax.add_collection3d(Line3DCollection(bonds, colors=(.55, .55, .55, .48), linewidths=.65))
    # Shaded ellipsoidal endcaps give actual oriented dumbbells with
    # vector PDF output. All surfaces share one collection for depth sorting.
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 17))
    sphere = np.stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)], axis=-1)
    faces, shades, rods = [], [], []
    light = np.array([-.35, -.45, 1.0])
    light /= np.linalg.norm(light)
    for point, state in zip(xyz, states.ravel()):
        direction = np.eye(3)[state]
        rods.append((point - .19 * direction, point + .19 * direction))
        radii = np.full(3, .105)
        radii[state] = .18
        for sign in (-1, 1):
            vertices = point + sign * .19 * direction + sphere * radii
            for i in range(sphere.shape[0] - 1):
                for j in range(sphere.shape[1] - 1):
                    face = vertices[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]]
                    normal = sphere[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]].mean(axis=0) / radii
                    normal /= np.linalg.norm(normal)
                    brightness = max(0, float(normal @ light))
                    color = COLORS[state] * (.50 + .45 * brightness) + .20 * brightness ** 12
                    faces.append(face)
                    shades.append(np.clip(color, 0, 1))
    ax.add_collection3d(Line3DCollection(rods, colors=COLORS[states.ravel()], linewidths=1.2))
    ax.add_collection3d(Poly3DCollection(faces, facecolors=shades, edgecolors="none", linewidths=0, antialiased=False))
    ax.set(xlim=(-.7, size - .3), ylim=(-.7, size - .3), zlim=(-.7, size - .3))
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-58)
    ax.set_axis_off()
    # Coordinate triad outside the crop, in the same world-coordinate camera.
    origin = np.array([-1.15, -1.15, -.80])
    for axis, label in enumerate("xyz"):
        vector = np.eye(3)[axis] * .85
        ax.quiver(*origin, *vector, color=COLORS[axis], linewidth=1.2, arrow_length_ratio=.18)
        ax.text(*(origin + 1.2 * vector), f"${label}$", color=COLORS[axis], fontsize=12)


def render(frames, sweeps, provenance, normal, index, origin, size, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    length = provenance["length"]
    if not 0 <= index < length:
        raise ValueError("Slice index must lie in [0,L)")
    if any(not 0 <= coordinate < length for coordinate in origin):
        raise ValueError("Inset origin coordinates must lie in [0,L)")
    if (index - origin[normal]) % length >= size:
        raise ValueError("The 3D ROI must intersect the selected slice")
    crop, xyz = periodic_crop(frames[-1], origin, size)
    prefix = Path(prefix).resolve()
    paths = [Path(str(prefix) + suffix) for suffix in
             ("_slice_triptych.png", "_slice_triptych.pdf", "_late_inset.png", "_late_inset.pdf", "_lattice.npz")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh prefix")
        for source in (Path(provenance["campaign"]), Path(provenance["trajectory"]).parent):
            if path == source or source in path.parents:
                raise ValueError("Output must be outside campaign and trial inputs")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    title = (f"{'GNN' if provenance['track'] == 'ml' else 'ED'}   $L={length}$   trial {provenance['trial']:03d}"
             f"   ${'xyz'[normal]}={index}$")
    style = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
             "axes.formatter.use_mathtext": True, "font.size": 11, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.65))
        fig.subplots_adjust(left=.055, right=.985, bottom=.13, top=.80, wspace=.22)
        slices = []
        for ax, frame, sweep in zip(axes, frames, sweeps):
            labels, plane = _slice(ax, frame, normal, index)
            slices.append(labels)
            ax.set_title(f"$t={sweep}$", pad=8)
        _mark_roi(axes[-1], origin, size, plane, length)
        fig.text(.055, .94, title, va="center")
        for state, label in enumerate("XYZ"):
            fig.text(.84 + .045 * state, .94, f"${label}$", color=COLORS[state], va="center")
        for path in paths[:2]:
            fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)

        fig = plt.figure(figsize=(7.5, 4.2))
        ax = fig.add_axes([.07, .13, .39, .70])
        _, plane = _slice(ax, frames[-1], normal, index)
        center = _mark_roi(ax, origin, size, plane, length)
        ax.set_title(f"$t={sweeps[-1]}$", pad=10)
        inset = fig.add_axes([.49, .03, .50, .86], projection="3d")
        _inset(inset, crop)
        inset.set_title(f"${size}\\times{size}\\times{size}$", y=.96, pad=0)
        center = (center[0] + min(size, length - origin[plane[0]]) / 2, center[1])
        fig.add_artist(ConnectionPatch(xyA=center, coordsA=ax.transData, xyB=(.19, .50),
                       coordsB=inset.transAxes, color="0.55", linewidth=.8,
                       linestyle=(0, (3, 3)), zorder=6))
        fig.text(.07, .95, title, va="center")
        for state, label in enumerate("XYZ"):
            fig.text(.81 + .055 * state, .95, f"${label}$", color=COLORS[state], va="center")
        for path in paths[2:4]:
            fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
    np.savez_compressed(paths[-1], **provenance, sweeps=np.asarray(sweeps), states=frames,
                        slice_states=np.stack(slices), slice_axis=np.array("xyz"[normal]), slice_index=index,
                        slice_plane_axes=np.array(plane), inset_origin=np.asarray(origin), inset_size=size,
                        roi_states=crop, roi_xyz=xyz, roi_site_ids=xyz[..., 0] * length ** 2 + xyz[..., 1] * length + xyz[..., 2],
                        colors=COLORS, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        coordinate_convention="states[x,y,z]; slice[first_plane_axis,second_plane_axis]; C-order z fastest",
                        inset_convention="Periodic unwrapped crop; gray NN rods illustrate geometry, not Hamiltonian bonds")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True, help="Campaign folder or campaign.json")
    parser.add_argument("--trial-dir", type=Path, required=True, help="Actual ML or ED trial_NNN folder")
    parser.add_argument("--sweeps", default="0,100,800", help="Exactly three increasing saved sweeps, comma separated")
    parser.add_argument("--slice-axis", choices=tuple("xyz"), default="z")
    parser.add_argument("--slice-index", type=int, help="Default: L//2")
    parser.add_argument("--inset-origin", help="Periodic crop origin x,y,z; default: centered on the slice")
    parser.add_argument("--inset-size", type=int, choices=(4, 6), default=4)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sweeps = [int(value) for value in args.sweeps.split(",")]
        frames, provenance = read_snapshots(args.campaign, args.trial_dir, sweeps)
        length, normal = provenance["length"], "xyz".index(args.slice_axis)
        index = length // 2 if args.slice_index is None else args.slice_index
        origin = [(length - args.inset_size) // 2] * 3
        origin[normal] = (index - args.inset_size // 2) % length
        if args.inset_origin:
            origin = [int(value) for value in args.inset_origin.split(",")]
        if len(origin) != 3:
            raise ValueError("--inset-origin requires three integers x,y,z")
        paths = render(frames, sweeps, provenance, normal, index, origin, args.inset_size, args.output_prefix)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
