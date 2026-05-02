from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def knn_accuracy(embeddings, labels, k=5):
    X = np.asarray(embeddings)
    y = np.asarray(labels)
    n = len(X)
    if n < 2:
        return 0.0
    correct = 0
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        if np.unique(y[train_mask]).size < 1:
            continue
        clf = KNeighborsClassifier(n_neighbors=min(k, n - 1), metric="cosine")
        clf.fit(X[train_mask], y[train_mask])
        pred = clf.predict(X[i : i + 1])[0]
        correct += int(pred == y[i])
    return correct / max(1, n)


def adjusted_rand_index(embeddings, labels, n_clusters):
    X = np.asarray(embeddings)
    n_samples = X.shape[0]
    if n_samples < 2:
        return 0.0
    n_clusters = max(1, min(int(n_clusters), n_samples))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(X)
    return adjusted_rand_score(labels, pred)


def mean_average_precision(embeddings, labels, k=10):
    X = np.asarray(embeddings)
    y = np.asarray(labels)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)

    ap_scores = []
    for i in range(X.shape[0]):
        idx = np.argsort(-sims[i])[:k]
        hits = 0
        precisions = []
        for rank, j in enumerate(idx, start=1):
            if y[j] == y[i]:
                hits += 1
                precisions.append(hits / rank)
        ap_scores.append(float(np.mean(precisions)) if precisions else 0.0)
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def linear_probe_accuracy(embeddings, labels, cv=5):
    X = np.asarray(embeddings)
    y = np.asarray(labels)
    encoded = LabelEncoder().fit_transform(y)
    if np.unique(encoded).size < 2:
        return 0.0
    splits = min(cv, np.min(np.bincount(encoded)))
    splits = max(2, int(splits))
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
        return float(np.mean(scores))
    except Exception:
        return 0.0


def compute_metrics(embeddings, labels, n_clusters, k=5, retrieval_k=10, linear_cv=5):
    X = np.asarray(embeddings)
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 and len(X) > 2 else 0.0
    return {
        "silhouette": sil,
        "knn_accuracy": float(knn_accuracy(X, labels, k=k)),
        "ari": float(adjusted_rand_index(X, labels, n_clusters=n_clusters)),
        "map": float(mean_average_precision(X, labels, k=retrieval_k)),
        "linear_probe": float(linear_probe_accuracy(X, labels, cv=linear_cv)),
    }


def _family_palette(families: list[str], family_colors: dict[str, str] | None):
    if family_colors is None:
        pal = sns.color_palette("tab10", n_colors=len(set(families)))
        uniq = sorted(set(families))
        return {f: pal[i] for i, f in enumerate(uniq)}
    return family_colors


def _plot_umap(ax, embeddings, labels, title, palette, subtitle_metric=None):
    emb = np.asarray(embeddings)
    if emb.shape[0] < 3:
        proj = np.zeros((emb.shape[0], 2), dtype=float)
        if emb.shape[1] >= 2:
            proj[:, :2] = emb[:, :2]
        elif emb.shape[1] == 1:
            proj[:, 0] = emb[:, 0]
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(embeddings)
    labels_arr = np.asarray(labels)
    for fam in sorted(set(labels)):
        m = labels_arr == fam
        ax.scatter(proj[m, 0], proj[m, 1], s=18, alpha=0.8, c=[palette[fam]], label=fam)
    sub = f"\nSilhouette: {subtitle_metric:.3f}" if subtitle_metric is not None else ""
    ax.set_title(f"{title}{sub}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def create_demo1_umap(
    method_embeddings: dict[str, np.ndarray],
    labels: list[str],
    metrics_by_method: dict[str, dict[str, float]],
    output_path: str | Path,
    family_colors: dict[str, str] | None = None,
    dpi: int = 300,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = _family_palette(labels, family_colors)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    method_order = [
        "gnn",
        "bag_of_node_types",
        "layer_histogram",
        "graph_statistics",
        "pca_adjacency",
    ]
    titles = {
        "gnn": "GNN Embeddings",
        "bag_of_node_types": "Bag of Node Types",
        "layer_histogram": "Layer Histogram",
        "graph_statistics": "Graph Statistics",
        "pca_adjacency": "PCA Adjacency",
    }

    for i, m in enumerate(method_order):
        _plot_umap(
            axes[i],
            method_embeddings[m],
            labels,
            titles[m],
            palette,
            subtitle_metric=metrics_by_method[m]["silhouette"],
        )

    ax_bar = axes[5]
    keys = ["knn_accuracy", "silhouette", "linear_probe", "map"]
    x = np.arange(len(keys))
    width = 0.15
    for i, m in enumerate(method_order):
        vals = [metrics_by_method[m][k] for k in keys]
        ax_bar.bar(x + (i - 2) * width, vals, width=width, label=titles[m])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["k-NN", "Silhouette", "Linear Probe", "Retrieval mAP"])
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Metrics Comparison")
    ax_bar.legend(fontsize=7)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette[f], label=f) for f in sorted(set(labels))]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_demo2_retrieval(
    embeddings: np.ndarray,
    prompts: list[str],
    labels: list[str],
    output_path: str | Path,
    k: int = 5,
    dpi: int = 300,
):
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)

    families = sorted(set(labels))
    query_indices = []
    for fam in families[:3]:
        query_indices.append(next(i for i, y in enumerate(labels) if y == fam))

    rows = []
    for q_idx in query_indices:
        neigh = np.argsort(-sims[q_idx])[:k]
        rows.append((q_idx, neigh))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    y0 = 0.95
    line_h = 0.08
    ax.text(0.01, y0, "Demo 2: Retrieval Examples (Top-5 cosine neighbors)", fontsize=14, weight="bold")
    y = y0 - 0.1
    for q_idx, neigh in rows:
        ax.text(
            0.01,
            y,
            f"Query [{labels[q_idx]}]: {prompts[q_idx]}",
            fontsize=10,
            weight="bold",
        )
        y -= line_h * 0.75
        for rank, n_idx in enumerate(neigh, start=1):
            ax.text(
                0.03,
                y,
                f"{rank}. ({labels[n_idx]}) sim={sims[q_idx, n_idx]:.3f} | {prompts[n_idx]}",
                fontsize=9,
            )
            y -= line_h * 0.65
        y -= line_h * 0.25
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_demo3_cluster_motifs(
    embeddings: np.ndarray,
    graphs: list[Any],
    labels: list[str],
    output_png: str | Path,
    output_txt: str | Path,
    kmeans_k: int = 8,
    dpi: int = 300,
):
    n_clusters = max(1, min(int(kmeans_k), embeddings.shape[0]))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    lines = ["Cluster Motif Analysis", "======================", ""]
    cluster_summary = []
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        fam_counter = Counter(labels[i] for i in idx)
        common_family, common_count = fam_counter.most_common(1)[0]

        motif_counter = Counter()
        prompt_examples = []
        for i in idx:
            g = graphs[i]
            prompt_examples.append(getattr(g, "prompt", "")[:80])
            layer_arr = g.layer_index.cpu().numpy() if hasattr(g, "layer_index") else np.zeros(g.num_nodes)
            type_arr = g.node_type.cpu().numpy() if hasattr(g, "node_type") else np.argmax(g.x[:, :4].cpu().numpy(), axis=1)
            for l, t in zip(layer_arr.astype(int), type_arr.astype(int)):
                motif_counter[(int(l), int(t))] += 1

        top_motifs = motif_counter.most_common(5)
        motif_str = ", ".join([f"(L{l},T{t}) x{c}" for (l, t), c in top_motifs])
        examples = "; ".join(prompt_examples[:3])
        cluster_summary.append((cid, common_family, common_count / idx.size, motif_str, examples))

        lines.append(f"Cluster {cid}")
        lines.append(f"- Most common family: {common_family} ({common_count}/{idx.size})")
        lines.append(f"- Top motifs: {motif_str}")
        lines.append(f"- Example prompts: {examples}")
        lines.append("")

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title("Demo 3: Cluster Motif Analysis (k=8)", fontsize=14, weight="bold")
    y = 0.95
    for cid, fam, frac, motif, ex in cluster_summary[:8]:
        txt = f"C{cid}: family={fam} ({frac:.2f}) | motifs={motif}\nexamples: {ex}"
        ax.text(0.01, y, txt, fontsize=9, va="top")
        y -= 0.11
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def create_demo4_metrics(
    metrics_by_method: dict[str, dict[str, float]],
    output_path: str | Path,
    dpi: int = 300,
):
    methods = list(metrics_by_method.keys())
    keys = ["knn_accuracy", "silhouette", "linear_probe", "map"]
    pretty = {
        "knn_accuracy": "k-NN",
        "silhouette": "Silhouette",
        "linear_probe": "Linear Probe",
        "map": "Retrieval mAP",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(keys))
    width = 0.12
    for i, m in enumerate(methods):
        vals = [metrics_by_method[m][k] for k in keys]
        ax.bar(x + (i - (len(methods) - 1) / 2.0) * width, vals, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[k] for k in keys])
    ax.set_ylabel("Score")
    ax.set_title("Demo 4: Metrics Comparison Across Methods")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def run_full_evaluation(
    method_embeddings: dict[str, np.ndarray],
    graphs: list[Any],
    labels: list[str],
    prompts: list[str],
    output_dir: str | Path,
    cfg: dict[str, Any],
) -> dict[str, dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = cfg["evaluation"]
    vis_cfg = cfg["visualization"]

    metrics_by_method: dict[str, dict[str, float]] = {}
    for method, emb in method_embeddings.items():
        metrics_by_method[method] = compute_metrics(
            emb,
            labels,
            n_clusters=int(eval_cfg["kmeans_k"]),
            k=int(eval_cfg["knn_k"]),
            retrieval_k=int(eval_cfg["retrieval_k"]),
            linear_cv=int(eval_cfg["linear_probe_cv"]),
        )

    create_demo1_umap(
        method_embeddings=method_embeddings,
        labels=labels,
        metrics_by_method=metrics_by_method,
        output_path=output_dir / "demo1_umap.png",
        family_colors=vis_cfg.get("family_colors", None),
        dpi=int(vis_cfg.get("dpi", 300)),
    )
    create_demo2_retrieval(
        embeddings=method_embeddings["gnn"],
        prompts=prompts,
        labels=labels,
        output_path=output_dir / "demo2_retrieval.png",
        k=int(eval_cfg["retrieval_k"]),
        dpi=int(vis_cfg.get("dpi", 300)),
    )
    create_demo3_cluster_motifs(
        embeddings=method_embeddings["gnn"],
        graphs=graphs,
        labels=labels,
        output_png=output_dir / "demo3_cluster_motifs.png",
        output_txt=output_dir / "demo3_cluster_motifs.txt",
        kmeans_k=int(eval_cfg["kmeans_k"]),
        dpi=int(vis_cfg.get("dpi", 300)),
    )
    create_demo4_metrics(
        metrics_by_method=metrics_by_method,
        output_path=output_dir / "demo4_metrics.png",
        dpi=int(vis_cfg.get("dpi", 300)),
    )

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_by_method, f, indent=2)
    return metrics_by_method
