from __future__ import annotations

import random
from typing import Callable

import torch
from torch_geometric.data import Data


def _clone_data(data: Data) -> Data:
    return data.clone()


def _filter_nodes(data: Data, keep_nodes: torch.Tensor) -> Data:
    keep_nodes = keep_nodes.bool()
    new_data = _clone_data(data)
    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
    old_to_new[keep_nodes] = torch.arange(keep_nodes.sum())

    if data.edge_index.numel() > 0:
        src = data.edge_index[0]
        dst = data.edge_index[1]
        keep_edges = keep_nodes[src] & keep_nodes[dst]
        new_edge_index = data.edge_index[:, keep_edges]
        new_edge_index = old_to_new[new_edge_index]
        new_data.edge_index = new_edge_index
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[keep_edges]
    else:
        new_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            new_data.edge_attr = torch.zeros((0, data.edge_attr.shape[-1]), dtype=data.edge_attr.dtype)

    new_data.x = data.x[keep_nodes]
    if hasattr(data, "node_type"):
        new_data.node_type = data.node_type[keep_nodes]
    if hasattr(data, "layer_index"):
        new_data.layer_index = data.layer_index[keep_nodes]
    return new_data


def node_drop(data: Data, drop_rate: float = 0.1) -> Data:
    new_data = _clone_data(data)
    n = data.num_nodes
    if n <= 1:
        return new_data

    keep = torch.ones(n, dtype=torch.bool)
    rand_mask = torch.rand(n) < drop_rate
    keep[rand_mask] = False

    # Never drop logit nodes.
    if hasattr(data, "node_type"):
        keep[data.node_type == 3] = True
    if keep.sum() == 0:
        keep[torch.randint(0, n, (1,))] = True
    return _filter_nodes(new_data, keep)


def edge_perturb(data: Data, perturb_rate: float = 0.15) -> Data:
    new_data = _clone_data(data)
    e = data.edge_index.shape[1]
    if e == 0:
        return new_data

    keep_edges = torch.rand(e) > (perturb_rate / 2.0)
    new_data.edge_index = data.edge_index[:, keep_edges]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
        new_data.edge_attr = data.edge_attr[keep_edges].clone()
        noise_mask = torch.rand(new_data.edge_attr.shape[0]) < perturb_rate
        noise = torch.randn_like(new_data.edge_attr[:, 0]) * 0.05
        new_data.edge_attr[noise_mask, 0] += noise[noise_mask]
    return new_data


def error_node_drop(data: Data, drop_rate: float = 0.5) -> Data:
    new_data = _clone_data(data)
    if not hasattr(data, "node_type"):
        return new_data
    keep = torch.ones(data.num_nodes, dtype=torch.bool)
    error_idx = torch.where(data.node_type == 2)[0]
    if error_idx.numel() == 0:
        return new_data
    drop_mask = torch.rand(error_idx.numel()) < drop_rate
    keep[error_idx[drop_mask]] = False
    if keep.sum() == 0:
        keep[error_idx[0]] = True
    return _filter_nodes(new_data, keep)


def layer_edge_drop(data: Data, n_layers_to_drop: int = 2, total_layers: int = 26) -> Data:
    new_data = _clone_data(data)
    if data.edge_index.numel() == 0 or not hasattr(data, "layer_index"):
        return new_data

    candidate_layers = list(range(total_layers))
    n_pick = min(n_layers_to_drop, len(candidate_layers))
    dropped_layers = set(random.sample(candidate_layers, n_pick))

    src_layers = data.layer_index[data.edge_index[0]]
    keep_edges = torch.tensor([int(layer.item()) not in dropped_layers for layer in src_layers]).bool()
    new_data.edge_index = data.edge_index[:, keep_edges]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr[keep_edges]
    return new_data


def compose_augmentations(
    data: Data,
    aug_list: list[Callable[[Data], Data]],
    p: float = 0.5,
) -> Data:
    out = _clone_data(data)
    for aug in aug_list:
        if random.random() < p:
            out = aug(out)
    return out
