#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import bag_of_node_types, graph_statistics, layer_histogram, pca_adjacency
from src.evaluation import run_full_evaluation
from src.model import CircuitGINEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def _extract_gnn_embeddings(model, graphs, device):
    model.eval()
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, g = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            out.append(g.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.projection_head[-1].out_features))


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    graphs = torch.load(args.dataset_path, weights_only=False)
    labels = [getattr(g, "family", "unknown") for g in graphs]
    prompts = [getattr(g, "prompt", "") for g in graphs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircuitGINEncoder(**cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    method_embeddings = {
        "gnn": _extract_gnn_embeddings(model, graphs, device),
        "bag_of_node_types": bag_of_node_types(graphs),
        "layer_histogram": layer_histogram(graphs),
        "graph_statistics": graph_statistics(graphs),
        "pca_adjacency": pca_adjacency(graphs, n_components=int(cfg["evaluation"]["pca_components"])),
    }

    metrics = run_full_evaluation(
        method_embeddings=method_embeddings,
        graphs=graphs,
        labels=labels,
        prompts=prompts,
        output_dir=Path(args.output_dir),
        cfg=cfg,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
