from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def bag_of_node_types(graphs):
    out = []
    for g in graphs:
        if hasattr(g, "node_type"):
            t = g.node_type.cpu().numpy()
        else:
            t = np.argmax(g.x[:, :4].cpu().numpy(), axis=1)
        hist = np.bincount(t.astype(int), minlength=4).astype(float)
        out.append(hist)
    return np.stack(out, axis=0) if out else np.zeros((0, 4))


def layer_histogram(graphs, n_layers: int = 26):
    feats = []
    for g in graphs:
        layer_counts = np.zeros(n_layers, dtype=float)
        influence_sums = np.zeros(n_layers, dtype=float)
        layers = g.layer_index.cpu().numpy() if hasattr(g, "layer_index") else np.zeros(g.num_nodes)
        influence = g.x[:, 7].cpu().numpy()
        for li, inf in zip(layers.astype(int), influence):
            if 0 <= li < n_layers:
                layer_counts[li] += 1.0
                influence_sums[li] += abs(float(inf))
        feats.append(np.concatenate([layer_counts, influence_sums], axis=0))
    return np.stack(feats, axis=0) if feats else np.zeros((0, n_layers * 2))


def _weakly_connected_components(num_nodes: int, edge_index: np.ndarray) -> int:
    if num_nodes == 0:
        return 0
    adj = [[] for _ in range(num_nodes)]
    for s, t in edge_index.T:
        s, t = int(s), int(t)
        adj[s].append(t)
        adj[t].append(s)
    seen = np.zeros(num_nodes, dtype=bool)
    comps = 0
    for i in range(num_nodes):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def graph_statistics(graphs):
    out = []
    for g in graphs:
        n = int(g.num_nodes)
        e = int(g.edge_index.shape[1])
        density = e / max(1, n * max(1, n - 1))
        edge_np = g.edge_index.cpu().numpy() if e > 0 else np.zeros((2, 0), dtype=int)
        indeg = np.bincount(edge_np[1], minlength=n) if n > 0 else np.array([0.0])
        outdeg = np.bincount(edge_np[0], minlength=n) if n > 0 else np.array([0.0])
        comps = _weakly_connected_components(n, edge_np)

        weights = g.edge_attr[:, 0].cpu().numpy() if hasattr(g, "edge_attr") and e > 0 else np.array([0.0])
        w_mean = float(np.mean(weights))
        w_std = float(np.std(weights))
        w_skew = float(np.mean(((weights - w_mean) / (w_std + 1e-8)) ** 3))

        spectral_gap = 0.0
        if n > 1 and e > 0 and n <= 512:
            A = np.zeros((n, n), dtype=float)
            for (s, t), w in zip(edge_np.T, weights):
                A[int(s), int(t)] += float(abs(w))
            deg = A.sum(axis=1)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
            L = np.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt
            eigvals = np.linalg.eigvalsh(L)
            eigvals = np.sort(eigvals)
            if eigvals.shape[0] >= 2:
                spectral_gap = float(eigvals[-1] - eigvals[-2])
            else:
                spectral_gap = float(eigvals[-1])

        feats = np.array(
            [
                n,
                e,
                density,
                float(np.mean(indeg)) if indeg.size else 0.0,
                float(np.std(indeg)) if indeg.size else 0.0,
                float(np.mean(outdeg)) if outdeg.size else 0.0,
                float(np.std(outdeg)) if outdeg.size else 0.0,
                float(comps),
                w_mean,
                w_std,
                w_skew,
                spectral_gap,
            ],
            dtype=float,
        )
        out.append(feats)
    return np.stack(out, axis=0) if out else np.zeros((0, 12))


def pca_adjacency(graphs, n_components=50):
    if not graphs:
        return np.zeros((0, n_components))
    max_nodes = max(int(g.num_nodes) for g in graphs)
    vecs = []
    for g in graphs:
        n = int(g.num_nodes)
        A = np.zeros((max_nodes, max_nodes), dtype=float)
        if g.edge_index.numel() > 0:
            edge_np = g.edge_index.cpu().numpy()
            weights = (
                g.edge_attr[:, 0].cpu().numpy()
                if hasattr(g, "edge_attr") and g.edge_attr is not None
                else np.ones(edge_np.shape[1], dtype=float)
            )
            for (s, t), w in zip(edge_np.T, weights):
                if s < max_nodes and t < max_nodes:
                    A[int(s), int(t)] = float(w)
        vecs.append(A.reshape(-1))
    X = np.stack(vecs, axis=0)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    if n_comp <= 1:
        return X[:, :1]
    return PCA(n_components=n_comp, random_state=42).fit_transform(X)
