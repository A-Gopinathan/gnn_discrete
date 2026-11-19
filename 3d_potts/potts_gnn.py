import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class ConfigurationEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W_longitudinal = nn.Linear(hidden_dim, hidden_dim)
        self.W_transverse = nn.Linear(hidden_dim, hidden_dim)
        self.W_t_outward = nn.Linear(hidden_dim, hidden_dim)
        self.W_t_inward = nn.Linear(hidden_dim, hidden_dim)
        self.W_cross = nn.Linear(hidden_dim, hidden_dim)

        self.node_lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_3d, edge_index, edge_type):
        src, dest = edge_index
        kappa = edge_type

        out_buckets = []

        # Explicitly evaluate the geometry for all 3 possible states
        for c_target in [0, 1, 2]:
            c_dest_tensor = torch.full_like(kappa, c_target)

            aggregated_neighbor_messages = 0

            for c_prime in [0, 1, 2]:
                c_src_tensor = torch.full_like(kappa, c_prime)

                f_j_cprime = x_3d[src, c_prime, :]

                mask_0 = (
                    (c_src_tensor == kappa)
                    & (c_dest_tensor == kappa)
                )

                mask_1 = (
                    (c_src_tensor == c_dest_tensor)
                    & (c_src_tensor != kappa)
                )

                mask_2 = (
                    (c_src_tensor == kappa)
                    & (c_dest_tensor != kappa)
                )

                mask_3 = (
                    (c_src_tensor != kappa)
                    & (c_dest_tensor == kappa)
                )

                mask_4 = (
                    (c_src_tensor != c_dest_tensor)
                    & (c_src_tensor != kappa)
                    & (c_dest_tensor != kappa)
                )

                m0 = mask_0.unsqueeze(-1)
                m1 = mask_1.unsqueeze(-1)
                m2 = mask_2.unsqueeze(-1)
                m3 = mask_3.unsqueeze(-1)
                m4 = mask_4.unsqueeze(-1)

                msg_0 = self.W_longitudinal(f_j_cprime)
                msg_1 = self.W_transverse(f_j_cprime)
                msg_2 = self.W_t_outward(f_j_cprime)
                msg_3 = self.W_t_inward(f_j_cprime)
                msg_4 = self.W_cross(f_j_cprime)

                messages = torch.where(
                    m0,
                    msg_0,
                    torch.where(
                        m1,
                        msg_1,
                        torch.where(
                            m2,
                            msg_2,
                            torch.where(
                                m3,
                                msg_3,
                                msg_4,
                            ),
                        ),
                    ),
                )

                aggregated_neighbor_messages = (
                    aggregated_neighbor_messages + messages
                )

            node_neighbor_contribution = scatter(
                aggregated_neighbor_messages,
                dest,
                dim=0,
                dim_size=x_3d.size(0),
                reduce="add",
            )

            node_self_contribution = self.node_lin_self(
                x_3d[:, c_target, :]
            )

            new_x = torch.relu(
                node_self_contribution
                + node_neighbor_contribution
            )

            new_x = self.node_norm(
                new_x + x_3d[:, c_target, :]
            )

            out_buckets.append(new_x)

        return torch.stack(out_buckets, dim=1)


class PottsGNN(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(
            1,
            hidden_dim,
        )

        self.layers = nn.ModuleList(
            [
                ConfigurationEquivariantLayer(hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        raw_states = data.x.squeeze().long()

        edge_index = data.edge_index
        edge_type = data.edge_attr.squeeze().long()

        one_hot = torch.nn.functional.one_hot(
            raw_states,
            num_classes=3,
        ).float()

        # Shape:
        # (num_nodes, 3 states, hidden_dim)
        x_3d = self.input_proj(
            one_hot.unsqueeze(-1)
        )

        for layer in self.layers:
            x_3d = layer(
                x_3d,
                edge_index,
                edge_type,
            )

        # Energy-like output for each possible state
        raw_delta_energies = self.output_layer(
            x_3d
        ).squeeze(-1)

        # Energy corresponding to each node's current state
        current_state_delta_energies = (
            raw_delta_energies.gather(
                1,
                raw_states.unsqueeze(1),
            )
        )

        # Define the current state's transition energy as exactly zero
        delta_E_predictions = (
            raw_delta_energies
            - current_state_delta_energies
        )

        return delta_E_predictions