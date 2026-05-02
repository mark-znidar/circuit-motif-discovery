#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_converter import convert_graph_file_to_data, is_graph_json_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_layers = int(cfg["conversion"]["n_layers"])

    input_dir = Path(args.input_dir)
    raw_json_files = sorted(input_dir.rglob("*.json"))
    graph_files = [p for p in raw_json_files if is_graph_json_file(p)]
    ignored_json = len(raw_json_files) - len(graph_files)
    if not graph_files:
        graph_files = sorted(input_dir.rglob("*.pt"))
    elif ignored_json > 0:
        print(f"Ignoring {ignored_json} non-graph JSON files (metadata/summary).")

    dataset = []
    node_counts = []
    edge_counts = []
    for i, path in enumerate(tqdm(graph_files, desc="Converting graphs", unit="graph"), start=1):
        try:
            data = convert_graph_file_to_data(path, graph_id=path.stem, n_layers=n_layers)
            dataset.append(data)
            node_counts.append(int(data.num_nodes))
            edge_counts.append(int(data.edge_index.shape[1]))
            print(
                f"Converting graph {i}/{len(graph_files)} | "
                f"Nodes: {min(node_counts)}-{max(node_counts)} (avg {sum(node_counts)/len(node_counts):.0f}) | "
                f"Edges: {min(edge_counts)}-{max(edge_counts)} (avg {sum(edge_counts)/len(edge_counts):.0f})"
            )
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    print(f"Saved {len(dataset)} graphs to {out}")


if __name__ == "__main__":
    main()
