#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.contrastive import train_contrastive
from src.model import CircuitGINEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    dataset = torch.load(args.dataset_path, weights_only=False)
    model = CircuitGINEncoder(**cfg["model"])
    train_contrastive(model, dataset, cfg, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
