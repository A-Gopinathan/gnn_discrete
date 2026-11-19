"""Local, Z2-invariant and D4-equivariant square-lattice Ising models.

Write q[i,d] = s[i] s[i+d], e[0,i,d] = encoder(q[i,d]), f[0,i] = 0.
Each isotropic layer is the recovered node <-> edge recurrence::

    u[i,d] = A e[i,d] + B (f[i] + f[i+d])
    e'[i,d] = LN_e(relu(u[i,d]) + residual * e[i,d])
    v[i] = C f[i] + sum_d D e'[i,d]
    f'[i] = LN_f(relu(v[i]) + residual * f[i])

Affine maps and normalizations are shared across sites/directions; each LN
acts only on the hidden features of ONE node or directed edge. The directional
variant adds ``A_reverse e[i,-d] + A_perp sum_{c perpendicular d} e[i,c]``
to u. The self, reverse and perpendicular relations are preserved by every D4
operation. The scalar readout is affine in final or concatenated tapped node
features (always including the final layer). Only products
of input spins enter, so global reversal invariance is exact by construction.

Receptive-radius proof: e[0,i,d] depends on the two bond endpoints, while f[0]
is constant. After layer 1, both e[1,i,d] and f[1,i] depend only on spins within
Manhattan distance 1 of i. If this radius is k-1, same-origin edge mixing does
not increase it, and f[k-1,i+d] extends it by at most one. Thus layer k has
radius k, with paths reaching that bound for generic nondegenerate weights.
This holds with either residual setting. No global pooling/normalization is
permitted. A (2R+1)-square patch therefore gives exactly the center prediction:
artificial periodic wrap edges can only enter from distance R+1, beyond its
dependency cone. The patch's physical indices are themselves periodic.

The isotropic architecture cannot distinguish one-site alternating stripes
from alternating 2x2 blocks on periodic L divisible by four. At each layer
all nodes remain identical and edges have just two features indexed by their
initial bond sign, since each node sees two bonds of each sign. Directional
mixing removes this particular collision; it does not guarantee ED accuracy.
"""

import math
import hashlib
from pathlib import Path
from collections.abc import Mapping
from numbers import Real

import torch
from torch import nn


_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CONFIG_KEYS = {"num_layers", "hidden_dim", "architecture", "residual", "readout"}


def _neighbors(features):
    return torch.stack(
        [torch.roll(features, shifts=(-dx, -dy), dims=(1, 2))
         for dx, dy in _DIRECTIONS],
        dim=-2,
    )


def _positive_scale(value):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError("target_scale must be a finite positive scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("target_scale must be a finite positive scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("target_scale must be a finite positive scalar")
    return value


class _InteractionLayer(nn.Module):
    def __init__(self, hidden_dim, architecture, residual):
        super().__init__()
        self.residual = residual
        self.edge_lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.edge_lin_cross = nn.Linear(hidden_dim, hidden_dim)
        self.node_lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.node_lin_cross = nn.Linear(hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        if architecture == "directional":
            self.edge_lin_reverse = nn.Linear(hidden_dim, hidden_dim)
            self.edge_lin_perp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, nodes, edges):
        edge_update = self.edge_lin_self(edges) + self.edge_lin_cross(
            nodes.unsqueeze(-2) + _neighbors(nodes)
        )
        if hasattr(self, "edge_lin_reverse"):
            edge_update = (
                edge_update
                + self.edge_lin_reverse(edges[..., [1, 0, 3, 2], :])
                + self.edge_lin_perp(
                    edges[..., [2, 2, 0, 0], :] + edges[..., [3, 3, 1, 1], :]
                )
            )
        new_edges = torch.relu(edge_update)
        if self.residual:
            new_edges = new_edges + edges
        new_edges = self.edge_norm(new_edges)
        new_nodes = torch.relu(
            self.node_lin_self(nodes) + self.node_lin_cross(new_edges).sum(dim=-2)
        )
        if self.residual:
            new_nodes = new_nodes + nodes
        return self.node_norm(new_nodes), new_edges


class IsingGNN(nn.Module):
    """Return normalized total single-flip energies of shape [B,L,L].

    Multiply outputs by the checkpoint's target_scale for physical energies.
    No training artifacts or torch_geometric dependency are required.

    Multiscale readout concatenates node features at sorted unique layers
    {1, max(1, L//3), max(1, 2*L//3), L}. Initialization first creates the
    baseline Linear(H, 1) at its original RNG point, then a wider linear
    readout with zero earlier-tap weights and the baseline weight/bias at the
    last tap. Same-seed final/multiscale models therefore share the backbone
    and initial physical function; earlier-tap weights remain trainable.
    """

    def __init__(self, num_layers=10, hidden_dim=64, architecture="isotropic", residual=True,
                 readout="final"):
        super().__init__()
        for name, value in (("num_layers", num_layers), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if architecture not in ("isotropic", "directional"):
            raise ValueError("architecture must be 'isotropic' or 'directional'")
        if not isinstance(residual, bool):
            raise ValueError("residual must be boolean")
        if readout not in ("final", "multiscale"):
            raise ValueError("readout must be 'final' or 'multiscale'")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.residual = residual
        self.readout = readout
        self.readout_layers = ((num_layers,) if readout == "final" else tuple(sorted({
            1, max(1, num_layers // 3), max(1, 2 * num_layers // 3), num_layers,
        })))
        self.receptive_radius = num_layers
        self.edge_encoder = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [_InteractionLayer(hidden_dim, architecture, residual) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        if readout == "multiscale":
            base_readout = self.output_layer
            self.output_layer = nn.Linear(len(self.readout_layers) * hidden_dim, 1)
            with torch.no_grad():
                self.output_layer.weight.zero_()
                self.output_layer.weight[:, -hidden_dim:].copy_(base_readout.weight)
                self.output_layer.bias.copy_(base_readout.bias)

    @property
    def model_config(self):
        """Fresh dictionary containing exactly the constructor arguments."""
        return {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "residual": self.residual,
            "readout": self.readout,
        }

    def forward(self, spins):
        if not isinstance(spins, torch.Tensor):
            raise TypeError("spins must be a torch.Tensor of shape [B,L,L]")
        if (spins.ndim != 3 or spins.shape[0] < 1 or spins.shape[1] < 1
                or spins.shape[1] != spins.shape[2]):
            raise ValueError("spins must have nonempty square shape [B,L,L]")
        if spins.is_complex() or spins.dtype == torch.bool or not torch.all((spins == 1) | (spins == -1)):
            raise ValueError("spins must contain only real +/-1 values")
        spins = spins.to(device=self.edge_encoder.weight.device, dtype=self.edge_encoder.weight.dtype)
        values = spins.unsqueeze(-1)
        edges = self.edge_encoder(values.unsqueeze(-2) * _neighbors(values))
        nodes = spins.new_zeros((*spins.shape, self.hidden_dim))
        tapped = []
        for index, layer in enumerate(self.layers, start=1):
            nodes, edges = layer(nodes, edges)
            if index in self.readout_layers:
                tapped.append(nodes)
        features = tapped[0] if len(tapped) == 1 else torch.cat(tapped, dim=-1)
        return self.output_layer(features).squeeze(-1)


def _validate_finite_tensors(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("checkpoint contains non-finite tensors")
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_tensors(item)


def load_checkpoint(path, device="cpu"):
    """Load the fresh pipeline schema strictly, not deprecated .pth artifacts.

    Physics is required as a metadata mapping, but its scientific convention
    must be checked by the calling physics workflow. All stored tensors,
    including any optimizer tensors, are checked for finiteness.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_config", "model_state_dict", "target_scale", "physics"}
    if not isinstance(checkpoint, Mapping) or not required.issubset(checkpoint):
        raise ValueError(f"checkpoint must contain {sorted(required)}")
    if "identity" in checkpoint:
        expected = checkpoint["identity"].get("source_sha256", {}).get("models.py")
        actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("Checkpoint model-source identity differs; use its archived implementation")
    config = checkpoint["model_config"]
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError("model_config must contain exactly the IsingGNN constructor kwargs")
    if not isinstance(checkpoint["physics"], Mapping):
        raise ValueError("checkpoint physics must be a metadata mapping")
    _positive_scale(checkpoint["target_scale"])
    _validate_finite_tensors(checkpoint)
    model = IsingGNN(**config)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("model_state_dict must be a tensor mapping")
    # Preserve precision rather than silently downcasting a double checkpoint.
    weight = state.get("edge_encoder.weight")
    if weight is not None:
        if not weight.is_floating_point():
            raise ValueError("model weights must be floating-point tensors")
        if any(v.dtype != weight.dtype for v in state.values()):
            raise ValueError("model weights must have a consistent floating-point dtype")
        model = model.to(dtype=weight.dtype)
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval()
    return model, checkpoint


class SelectedSitePredictor:
    """Physical single-site energies with work independent of lattice length.

    Sites are flat row-major indices (x*L+y). Only the extracted patches are
    transferred to the model device and checked for valid spins. Callers must
    validate a complete input lattice once when initializing their simulation,
    not on every attempted update. Tensor and NumPy inputs are accepted.
    L <= 2R is explicitly rejected: no hidden full-lattice fallback exists.
    """

    def __init__(self, model, length, target_scale):
        if not isinstance(model, IsingGNN):
            raise TypeError("model must be an IsingGNN")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 2 * model.receptive_radius:
            raise ValueError("length must be an integer greater than 2*receptive_radius")
        self.model = model
        self.length = length
        self.target_scale = _positive_scale(target_scale)
        self.receptive_radius = model.receptive_radius
        self.patch_size = 2 * self.receptive_radius + 1
        self._offsets = torch.arange(-self.receptive_radius, self.receptive_radius + 1)

    @torch.inference_mode()
    def __call__(self, states, sites):
        states = torch.as_tensor(states)
        if (states.ndim != 3 or states.shape[0] < 1
                or tuple(states.shape[1:]) != (self.length, self.length)):
            raise ValueError("states must have shape [B,length,length]")
        sites = torch.as_tensor(sites, device=states.device)
        if sites.ndim != 1 or sites.shape[0] != states.shape[0]:
            raise ValueError("sites must have shape [B]")
        if sites.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("sites must contain integer indices")
        sites = sites.to(dtype=torch.long)
        if torch.any((sites < 0) | (sites >= self.length ** 2)):
            raise ValueError("sites are outside the lattice")
        offsets = self._offsets.to(device=states.device)
        rows = (sites[:, None] // self.length + offsets) % self.length
        cols = (sites[:, None] % self.length + offsets) % self.length
        batch = torch.arange(states.shape[0], device=states.device)[:, None, None]
        patches = states[batch, rows[:, :, None], cols[:, None, :]]
        predictions = self.model(patches)
        radius = self.receptive_radius
        return predictions[:, radius, radius] * self.target_scale
