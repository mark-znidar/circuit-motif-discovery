from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from circuit_tracer.utils.create_graph_files import create_graph_files


NODE_TYPES = {"embedding": 0, "feature": 1, "error": 2, "logit": 3}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _infer_node_type(feature_type: str | None) -> str:
    text = (feature_type or "").lower()
    if "logit" in text:
        return "logit"
    if "embedding" in text:
        return "embedding"
    if "error" in text or "reconstruction" in text:
        return "error"
    if "transcoder" in text:
        return "feature"
    return "feature"


def _node_to_feature_vector(
    node: dict[str, Any],
    max_ctx_idx: int,
    n_layers: int = 26,
) -> tuple[list[float], int, int]:
    feature_type = _infer_node_type(node.get("feature_type"))
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[NODE_TYPES[feature_type]] = 1.0

    layer_raw = node.get("layer", 0)
    if isinstance(layer_raw, str):
        layer_idx = n_layers if layer_raw.upper() == "E" else _safe_int(layer_raw, 0)
    else:
        layer_idx = _safe_int(layer_raw, 0)
    layer_norm = layer_idx / float(n_layers)

    ctx_idx = _safe_int(node.get("ctx_idx"), 0)
    pos_norm = ctx_idx / float(max(1, max_ctx_idx))

    act = _safe_float(node.get("activation"), 0.0)
    act_log = math.copysign(math.log1p(abs(act)), act)

    influence = _safe_float(node.get("influence"), 0.0)
    vec = one_hot + [layer_norm, pos_norm, act_log, influence]
    return vec, layer_idx, NODE_TYPES[feature_type]


def _load_json_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pt_to_json_payload(pt_path: Path) -> dict[str, Any]:
    # Robust fallback: convert .pt graph to JSON using circuit-tracer utility.
    with tempfile.TemporaryDirectory() as tmpdir:
        create_graph_files(
            graph_or_path=str(pt_path),
            slug="converted_graph",
            output_path=tmpdir,
            node_threshold=0.8,
            edge_threshold=0.98,
        )
        json_path = Path(tmpdir) / "converted_graph.json"
        return _load_json_payload(json_path)


def convert_json_payload_to_data(
    payload: dict[str, Any],
    family_label: str | None = None,
    graph_id: str | None = None,
    n_layers: int = 26,
) -> Data:
    nodes = payload.get("nodes", []) or []
    links = payload.get("links", []) or []
    metadata = payload.get("metadata", {}) or {}

    node_ids = [str(node.get("node_id", f"node_{i}")) for i, node in enumerate(nodes)]
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    max_ctx = max((_safe_int(node.get("ctx_idx"), 0) for node in nodes), default=1)

    x_rows: list[list[float]] = []
    layer_index: list[int] = []
    node_type_index: list[int] = []
    for node in nodes:
        vec, layer_idx, type_idx = _node_to_feature_vector(node, max_ctx_idx=max_ctx, n_layers=n_layers)
        x_rows.append(vec)
        layer_index.append(layer_idx)
        node_type_index.append(type_idx)

    if not x_rows:
        x = torch.zeros((1, 8), dtype=torch.float32)
        layer_tensor = torch.zeros((1,), dtype=torch.long)
        node_type_tensor = torch.tensor([NODE_TYPES["feature"]], dtype=torch.long)
    else:
        x = torch.tensor(x_rows, dtype=torch.float32)
        layer_tensor = torch.tensor(layer_index, dtype=torch.long)
        node_type_tensor = torch.tensor(node_type_index, dtype=torch.long)

    edge_pairs: list[list[int]] = []
    edge_weights: list[list[float]] = []
    for link in links:
        src = str(link.get("source", ""))
        dst = str(link.get("target", ""))
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        edge_pairs.append([node_to_idx[src], node_to_idx[dst]])
        edge_weights.append([_safe_float(link.get("weight"), 0.0)])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        layer_index=layer_tensor,
        node_type=node_type_tensor,
    )
    data.prompt = metadata.get("prompt", "")
    data.family = family_label or metadata.get("family", "unknown")
    data.graph_id = graph_id or metadata.get("slug", "graph")
    return data


def convert_graph_file_to_data(
    graph_path: str | Path,
    family_label: str | None = None,
    graph_id: str | None = None,
    n_layers: int = 26,
) -> Data:
    graph_path = Path(graph_path)
    if graph_path.suffix == ".json":
        payload = _load_json_payload(graph_path)
    elif graph_path.suffix == ".pt":
        payload = _pt_to_json_payload(graph_path)
    else:
        raise ValueError(f"Unsupported graph format: {graph_path}")
    return convert_json_payload_to_data(payload, family_label=family_label, graph_id=graph_id, n_layers=n_layers)


def convert_directory_to_dataset(input_dir: str | Path, n_layers: int = 26) -> list[Data]:
    input_dir = Path(input_dir)
    graph_files = sorted(input_dir.rglob("*.json")) + sorted(input_dir.rglob("*.pt"))
    dataset: list[Data] = []
    for idx, graph_path in enumerate(graph_files):
        try:
            dataset.append(
                convert_graph_file_to_data(
                    graph_path,
                    graph_id=f"graph_{idx:05d}",
                    n_layers=n_layers,
                )
            )
        except Exception as exc:
            print(f"Skipping {graph_path}: {exc}")
    return dataset
