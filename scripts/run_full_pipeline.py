#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--prompt_file", type=str, default="prompts/prompt_corpus.json")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def run(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    with open(root / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_prompts = None
    epochs = None
    batch_size = None
    if args.quick:
        max_prompts = int(cfg["quick"]["max_prompts_per_family"])
        epochs = int(cfg["quick"]["epochs"])
        batch_size = int(cfg["quick"]["batch_size"])

    graphs_dir = Path(args.data_dir) / "graphs"
    dataset_path = Path(args.data_dir) / "circuit_dataset.pt"
    ckpt_path = Path(args.checkpoints_dir) / "best.pt"

    step1 = [
        "python",
        "scripts/01_generate_graphs.py",
        "--prompt_file",
        args.prompt_file,
        "--output_dir",
        str(graphs_dir),
        "--device",
        args.device,
        "--config",
        args.config,
    ]
    if max_prompts is not None:
        step1 += ["--max_prompts_per_family", str(max_prompts)]

    step2 = [
        "python",
        "scripts/02_convert_to_pyg.py",
        "--input_dir",
        str(graphs_dir),
        "--output_path",
        str(dataset_path),
        "--config",
        args.config,
    ]

    step3 = [
        "python",
        "scripts/03_train_contrastive.py",
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        args.checkpoints_dir,
        "--config",
        args.config,
    ]
    if epochs is not None:
        step3 += ["--epochs", str(epochs)]
    if batch_size is not None:
        step3 += ["--batch_size", str(batch_size)]

    step4 = [
        "python",
        "scripts/04_evaluate.py",
        "--dataset_path",
        str(dataset_path),
        "--checkpoint",
        str(ckpt_path),
        "--output_dir",
        args.results_dir,
        "--config",
        args.config,
    ]

    run(step1, cwd=root)
    run(step2, cwd=root)
    run(step3, cwd=root)
    run(step4, cwd=root)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
