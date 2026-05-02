from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool


class CircuitGINEncoder(torch.nn.Module):
    """3-layer GIN with edge features (GINEConv), sum pooling, projection head."""

    def __init__(
        self,
        in_dim=8,
        hidden_dim=128,
        out_dim=64,
        num_layers=3,
        edge_dim=1,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.dropout = float(dropout)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], self.edge_dim), device=x.device, dtype=x.dtype)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = torch.relu(h)
            if self.dropout > 0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        g = global_add_pool(h, batch)
        z = self.projection_head(g)
        return z, g

    @torch.no_grad()
    def get_embedding(self, data):
        self.eval()
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        _, g = self.forward(x, edge_index, edge_attr, batch)
        return g.squeeze(0)
