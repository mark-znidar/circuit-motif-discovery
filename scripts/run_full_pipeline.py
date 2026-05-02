#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
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
    p.add_argument("--hf_token", type=str, default=None, help="Optional HF token for gated model access")
    p.add_argument("--skip_brief", action="store_true", help="Skip post-run markdown brief generation")
    return p.parse_args()


def run(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _assert_nonempty_generation(graphs_dir: Path):
    summary = _load_json_if_exists(graphs_dir / "summary.json")
    if summary is None:
        return
    generated = int(summary.get("total_generated", 0))
    if generated <= 0:
        failed = int(summary.get("failed", 0))
        raise RuntimeError(
            "Graph generation produced zero usable graphs. "
            f"failed={failed}. Check Hugging Face auth/token and gated model access "
            "before continuing."
        )


def _assert_nonempty_dataset(dataset_path: Path):
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset file missing: {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)
    n = len(data) if hasattr(data, "__len__") else 0
    if n <= 0:
        raise RuntimeError(
            "Converted dataset is empty. Ensure graph conversion succeeded and input graph files are valid."
        )


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
    if args.hf_token:
        step1 += ["--hf_token", args.hf_token]

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
    _assert_nonempty_generation(root / graphs_dir)
    run(step2, cwd=root)
    _assert_nonempty_dataset(root / dataset_path)
    run(step3, cwd=root)
    run(step4, cwd=root)
    if not args.skip_brief:
        step5 = [
            "python",
            "scripts/05_generate_brief.py",
            "--results_dir",
            args.results_dir,
            "--graphs_dir",
            str(graphs_dir),
            "--output_path",
            str(Path(args.results_dir) / "research_brief.md"),
        ]
        run(step5, cwd=root)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
