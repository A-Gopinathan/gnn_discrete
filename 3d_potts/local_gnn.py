"""Exact selected-site PottsGNN inference using native torch only."""
import itertools
import math
from types import SimpleNamespace

import torch
from torch import nn


PHYSICS = dict(temperature=0.1, filling=0.5, alpha=1.0, beta=-0.5, gamma=-0.7)
RELATIONS = ("W_longitudinal", "W_transverse", "W_t_outward", "W_t_inward", "W_cross")


def offsets(radius):
    return sorted(p for p in itertools.product(range(-radius, radius + 1), repeat=3)
                  if sum(map(abs, p)) <= radius)


class LocalLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        for name in RELATIONS:
            setattr(self, name, nn.Linear(hidden_dim, hidden_dim))
        self.node_lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, self_indices, message_indices):
        # Apply each affine map once per source/channel, retaining every edge bias.
        transformed = torch.stack([getattr(self, n)(x) for n in RELATIONS], dim=3)
        flat = transformed.flatten(1, 3)
        messages = flat[:, message_indices]  # B,target,target_channel,neighbor,source_channel,H
        summed = (messages[..., 0, :] + messages[..., 1, :]) + messages[..., 2, :]
        neighbor = summed.sum(dim=3)
        own = x[:, self_indices]
        return self.node_norm(torch.relu(self.node_lin_self(own) + neighbor) + own)


class LocalPottsGNN(nn.Module):
    def __init__(self, length, num_layers=3, hidden_dim=64, target_scale=1.0):
        super().__init__()
        if type(length) is not int or type(num_layers) is not int or num_layers < 1 or length <= 2 * num_layers:
            raise ValueError("length must exceed twice the positive layer count")
        if hidden_dim < 1 or not math.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("invalid hidden dimension or target scale")
        self.length, self.num_layers, self.target_scale = length, num_layers, float(target_scale)
        self.input_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([LocalLayer(hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(),
                                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.patch_sizes = [len(offsets(r)) for r in range(num_layers, -1, -1)]
        xyz = torch.cartesian_prod(*[torch.arange(length)] * 3)
        patch = (xyz[:, None, :] + torch.tensor(offsets(num_layers))[None, :, :]) % length
        lookup = (patch[..., 0] * length + patch[..., 1]) * length + patch[..., 2]
        self.register_buffer("lookup", lookup, persistent=False)
        for layer_id, radius in enumerate(range(num_layers, 0, -1)):
            sources, targets = offsets(radius), offsets(radius - 1)
            index = {p: i for i, p in enumerate(sources)}
            own, gather = [], []
            for p in targets:
                own.append(index[p])
                channels = []
                for ct in range(3):
                    neighbors = []
                    for axis in range(3):
                        for sign in (-1, 1):
                            q = list(p)
                            q[axis] += sign
                            source = index[tuple(q)]
                            routes = []
                            for cs in range(3):
                                relation = (0 if cs == ct == axis else 1 if cs == ct else
                                            2 if cs == axis else 3 if ct == axis else 4)
                                routes.append((source * 3 + cs) * 5 + relation)
                            neighbors.append(routes)
                    channels.append(neighbors)
                gather.append(channels)
            self.register_buffer(f"own_{layer_id}", torch.tensor(own), persistent=False)
            self.register_buffer(f"gather_{layer_id}", torch.tensor(gather), persistent=False)

    def forward(self, states, sites):
        if states.ndim != 2 or states.shape[1] != self.length ** 3 or sites.shape != (states.shape[0],):
            raise ValueError("expected states[B,L**3] and sites[B]")
        selected = states.gather(1, self.lookup[sites]).long()
        x = self.input_proj(torch.nn.functional.one_hot(selected, 3).float().unsqueeze(-1))
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"own_{i}"), getattr(self, f"gather_{i}"))
        raw = self.output_layer(x[:, 0]).squeeze(-1)
        current = states.gather(1, sites[:, None]).long()
        return (raw - raw.gather(1, current)) * self.target_scale


def load_local(checkpoint, length, device):
    config = checkpoint["config"]
    if config.get("model") != {"num_layers": 3, "hidden_dim": 64}:
        raise ValueError("production checkpoint must use 3 layers and width 64")
    settings = config.get("settings", {})
    if settings.get("smoke") or settings.get("overfit") or config.get("selection_metric") == "train_mse":
        raise ValueError("smoke/overfit checkpoints are not production models")
    physics = config.get("metadata", {}).get("physics")
    if physics is not None and any(physics.get(k) != v for k, v in PHYSICS.items()):
        raise ValueError("checkpoint physics mismatch")
    model = LocalPottsGNN(length, target_scale=float(checkpoint["target_scale"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("nonfinite checkpoint parameters")
    return model.to(device).eval().requires_grad_(False)


def full_graph(states, length):
    """Reference graph with directed periodic neighbors, x/y/z edge types."""
    xyz = torch.cartesian_prod(*[torch.arange(length, device=states.device)] * 3)
    src, dst, axes = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            q = xyz.clone()
            q[:, axis] = (q[:, axis] + sign) % length
            src.append((q[:, 0] * length + q[:, 1]) * length + q[:, 2])
            dst.append(torch.arange(length ** 3, device=states.device))
            axes.append(torch.full((length ** 3,), axis, device=states.device))
    return SimpleNamespace(x=states[:, None], edge_index=torch.stack([torch.cat(src), torch.cat(dst)]),
                           edge_attr=torch.cat(axes))


@torch.inference_mode()
def verify_full(model, states, sites):
    """Compare actual selected sites with the original model on bounded graphs.

    Only the radius-R dependency cone can affect the center after R layers.
    Embed that cone without changing axes or orbital labels; the arbitrary
    exterior cannot reach the center. Reference memory is independent of L.
    """
    from potts_gnn import PottsGNN
    if states.shape != (model.length ** 3,) or sites.ndim != 1 or not len(sites):
        raise ValueError("expected states[L**3] and nonempty sites[K]")
    if sites.dtype != torch.long or torch.any((sites < 0) | (sites >= states.numel())):
        raise ValueError("sites must be valid long indices")
    radius = model.num_layers
    length = 2 * radius + 2  # Even, and strictly wider than the dependency cone.
    center = length // 2
    patch = torch.tensor(offsets(radius), device=states.device)
    embedded = (patch + center) % length
    indices = (embedded[:, 0] * length + embedded[:, 1]) * length + embedded[:, 2]
    center_index = (center * length + center) * length + center
    original = PottsGNN(model.num_layers, model.input_proj.out_features).to(states.device).eval()
    original.load_state_dict(model.state_dict())
    maximum = 0.0
    for site in sites:
        xyz = torch.stack((site // model.length ** 2,
                           (site // model.length) % model.length, site % model.length))
        source = (patch + xyz) % model.length
        source_indices = (source[:, 0] * model.length + source[:, 1]) * model.length + source[:, 2]
        small = states.new_zeros(length ** 3)
        small[indices] = states[source_indices]
        reference = original(full_graph(small, length))[center_index] * model.target_scale
        actual = model(states[None, :], site.reshape(1))[0]
        current = states[site].long()
        if actual[current] != 0 or reference[current] != 0:
            raise AssertionError("self-transition prediction must be exactly zero")
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        maximum = max(maximum, float((actual - reference).abs().max().cpu()))
    return maximum


def heatbath_sample(delta, uniforms, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probabilities = torch.softmax(-delta.double() / temperature, dim=-1)
    return (uniforms[:, None] >= probabilities.cumsum(-1)[:, :2]).sum(-1)
