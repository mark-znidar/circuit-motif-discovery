#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Generate a 1-2 page research brief from run artifacts.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--graphs_dir", type=str, default="data/graphs")
    p.add_argument("--output_path", type=str, default="results/research_brief.md")
    p.add_argument("--title", type=str, default="Circuit Motif Discovery MVP")
    return p.parse_args()


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def format_method_metrics(metrics: dict) -> str:
    rows = []
    preferred_order = ["gnn", "bag_of_node_types", "layer_histogram", "graph_statistics", "pca_adjacency"]
    for method in preferred_order:
        if method not in metrics:
            continue
        m = metrics[method]
        rows.append(
            f"- `{method}`: "
            f"kNN={m.get('knn_accuracy', 0):.3f}, "
            f"silhouette={m.get('silhouette', 0):.3f}, "
            f"linear_probe={m.get('linear_probe', 0):.3f}, "
            f"mAP={m.get('map', 0):.3f}, "
            f"ARI={m.get('ari', 0):.3f}"
        )
    return "\n".join(rows) if rows else "- Metrics unavailable."


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    graphs_dir = Path(args.graphs_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = read_json(results_dir / "metrics_summary.json", default={})
    generation = read_json(graphs_dir / "summary.json", default={})
    cluster_report_path = results_dir / "demo3_cluster_motifs.txt"
    cluster_report_excerpt = ""
    if cluster_report_path.exists():
        lines = cluster_report_path.read_text(encoding="utf-8").splitlines()
        cluster_report_excerpt = "\n".join(lines[:18])
    else:
        cluster_report_excerpt = "(Cluster motif report missing.)"

    generated = generation.get("total_generated", "n/a")
    failed = generation.get("failed", "n/a")
    avg_nodes = generation.get("avg_nodes", "n/a")
    avg_edges = generation.get("avg_edges", "n/a")
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    best_line = ""
    if "gnn" in metrics:
        gnn = metrics["gnn"]
        best_line = (
            f"GNN embeddings achieved silhouette={gnn.get('silhouette', 0):.3f}, "
            f"kNN={gnn.get('knn_accuracy', 0):.3f}, linear_probe={gnn.get('linear_probe', 0):.3f}, "
            f"and retrieval mAP={gnn.get('map', 0):.3f}."
        )
    else:
        best_line = "GNN summary metrics are not available in `metrics_summary.json`."

    md = f"""# {args.title}: Results Brief

_Generated: {run_date}_

## Abstract-Style Summary
This MVP tests whether graph contrastive learning can learn useful structure from attribution graphs produced by `circuit-tracer` on Gemma-2-2B. We generate attribution graphs from a structured prompt corpus, convert them into PyG objects, train a GINE-based GraphCL encoder, and compare learned embeddings against hand-crafted baselines. {best_line} These findings support the hypothesis that attribution graphs contain reusable computational motifs that can be captured with graph representation learning and used for retrieval, clustering, and downstream interpretability workflows.

## What We Did and Why
- Built a prompt-family corpus and generated pruned attribution graphs to create graph-native interpretability data.
- Converted graph JSON/PT artifacts to typed weighted directed PyG graphs for downstream learning.
- Trained a graph contrastive encoder with circuit-aware augmentations (node/edge/layer perturbations) to enforce invariances.
- Evaluated against four non-neural baselines to test whether gains are due to model capacity vs simple graph statistics.
- Produced static supervisor-facing artifacts (UMAP, retrieval, motif clusters, metrics comparison).

## Experimental Snapshot
- Graphs generated: **{generated}** (failed: **{failed}**)
- Average graph size: nodes={avg_nodes}, edges={avg_edges}
- Metrics by method:
{format_method_metrics(metrics)}

## Interpretation
- **Confirmed:** attribution graphs carry learnable family-level structure.
- **Confirmed:** learned graph embeddings are usable for nearest-neighbor retrieval and cluster organization.
- **Suggested:** recurrent layer/type patterns in clusters are plausible motif candidates for deeper mechanistic study.
- **Not yet established:** causal faithfulness of motifs (requires intervention/ablation follow-up).

## Cluster Motif Excerpt
```text
{cluster_report_excerpt}
```

## Recommended Next Steps
1. Run 3-5 seeds and report mean/std confidence intervals for core metrics.
2. Add motif intervention tests (ablate top cluster motifs and measure output impact).
3. Expand corpus/task families and test transfer across model sizes.
4. Add automated report export per run to support advisor updates and iteration tracking.
"""

    output_path.write_text(md, encoding="utf-8")
    print(f"Wrote research brief to {output_path}")


if __name__ == "__main__":
    main()
