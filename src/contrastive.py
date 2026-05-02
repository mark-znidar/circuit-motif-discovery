from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console
from torch_geometric.loader import DataLoader

from .augmentations import compose_augmentations, edge_perturb, error_node_drop, layer_edge_drop, node_drop

console = Console()


def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross-Entropy Loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.mm(z, z.t()) / temperature
    n = z1.size(0)
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)

    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    loss = F.cross_entropy(sim, labels)
    return loss


def _make_views(batch, cfg):
    aug_cfg = cfg["augmentations"]
    aug_fns = [
        lambda d: node_drop(d, drop_rate=aug_cfg["node_drop_rate"]),
        lambda d: edge_perturb(d, perturb_rate=aug_cfg["edge_perturb_rate"]),
        lambda d: error_node_drop(d, drop_rate=aug_cfg["error_node_drop_rate"]),
        lambda d: layer_edge_drop(d, n_layers_to_drop=aug_cfg["layer_drop_count"]),
    ]
    graphs = batch.to_data_list()
    v1 = [compose_augmentations(g, aug_fns, p=0.5) for g in graphs]
    v2 = [compose_augmentations(g, aug_fns, p=0.5) for g in graphs]
    return v1, v2


def train_contrastive(model, dataset, config: dict[str, Any], output_dir: str | Path = "checkpoints"):
    """
    GraphCL training loop with per-epoch progress reporting.
    """
    train_cfg = config["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_loss = float("inf")
    history: list[float] = []
    epochs = int(train_cfg["epochs"])
    ckpt_every = int(train_cfg.get("checkpoint_every", 50))
    start_all = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            view1_list, view2_list = _make_views(batch, config)
            b1 = next(iter(DataLoader(view1_list, batch_size=len(view1_list), shuffle=False))).to(device)
            b2 = next(iter(DataLoader(view2_list, batch_size=len(view2_list), shuffle=False))).to(device)

            z1, _ = model(b1.x, b1.edge_index, b1.edge_attr, b1.batch)
            z2, _ = model(b2.x, b2.edge_index, b2.edge_attr, b2.batch)
            loss = nt_xent_loss(z1, z2, temperature=float(train_cfg["temperature"]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        history.append(avg_loss)
        elapsed_epoch = time.time() - epoch_start
        eta = (epochs - epoch) * elapsed_epoch
        lr = optimizer.param_groups[0]["lr"]
        console.print(
            f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr:.2e} "
            f"| Time: {elapsed_epoch:.1f}s/epoch | ETA: {eta/60:.1f}m"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "history": history,
                    "config": config,
                },
                output_dir / "best.pt",
            )

        if epoch % ckpt_every == 0 or epoch == epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg_loss,
                    "history": history,
                    "config": config,
                },
                output_dir / f"epoch_{epoch:04d}.pt",
            )

    total_time = time.time() - start_all
    console.print(f"[bold green]Training complete[/bold green] in {total_time/60:.1f} minutes")
    return {"history": history, "best_loss": best_loss}
