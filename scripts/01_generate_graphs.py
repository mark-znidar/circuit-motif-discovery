#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.graph_generation import GenerationConfig, generate_graphs, load_prompt_corpus


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--families", type=str, default=None)
    parser.add_argument("--max_prompts_per_family", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen_cfg = GenerationConfig(**cfg["generation"])

    families = [f.strip() for f in args.families.split(",")] if args.families else None
    prompts = load_prompt_corpus(
        args.prompt_file,
        families=families,
        max_prompts_per_family=args.max_prompts_per_family,
    )
    summary = generate_graphs(prompts, args.output_dir, gen_cfg, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
