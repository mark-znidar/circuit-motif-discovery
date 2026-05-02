#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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


def infer_family_from_stem(stem: str) -> str | None:
    # Expected slug pattern: <family>_0001_<prompt_slug>
    match = re.match(r"^(?P<family>.+)_\d{4}_", stem)
    return match.group("family") if match else None


def infer_family_from_metadata(path: Path) -> str | None:
    # Graph JSON: data/graphs/json/<slug>.json
    # Metadata:   data/graphs/metadata/<slug>.meta.json
    slug = path.stem
    metadata_path = path.parents[1] / "metadata" / f"{slug}.meta.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    fam = payload.get("family", None)
    return str(fam) if fam else None


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
            family = infer_family_from_metadata(path) or infer_family_from_stem(path.stem) or "unknown"
            data = convert_graph_file_to_data(
                path,
                family_label=family,
                graph_id=path.stem,
                n_layers=n_layers,
            )
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
