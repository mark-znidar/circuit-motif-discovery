from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


class CircuitGraphDataset(InMemoryDataset):
    """Simple in-memory dataset wrapper for pre-converted PyG Data objects."""

    def __init__(self, data_list=None, path: str | Path | None = None):
        super().__init__(root=".")
        if path is not None:
            loaded = torch.load(path, weights_only=False)
            if isinstance(loaded, list):
                data_list = loaded
            elif isinstance(loaded, dict) and "data_list" in loaded:
                data_list = loaded["data_list"]
            else:
                raise ValueError(f"Unsupported dataset format in {path}")
        self.data_list = data_list or []
        self.data, self.slices = self.collate(self.data_list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data_list, path)
