from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm.auto import tqdm

console = Console()


@dataclass
class GenerationConfig:
    model: str = "google/gemma-2-2b"
    transcoder_set: str = "gemma"
    offload: str | None = "cpu"
    max_feature_nodes: int = 7500
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    dtype: str = "bfloat16"
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    batch_size: int = 256
    backend: str = "transformerlens"
    lazy_encoder: bool = False
    lazy_decoder: bool = True


def load_prompt_corpus(
    prompt_file: str | Path,
    families: list[str] | None = None,
    max_prompts_per_family: int | None = None,
) -> list[dict[str, str]]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    family_map = payload.get("families", {})
    selected_families = families or list(family_map.keys())
    prompts: list[dict[str, str]] = []

    for family in selected_families:
        if family not in family_map:
            console.print(f"[yellow]Skipping unknown family: {family}[/yellow]")
            continue
        family_prompts = family_map[family].get("prompts", [])
        if max_prompts_per_family is not None:
            family_prompts = family_prompts[:max_prompts_per_family]
        prompts.extend({"family": family, "prompt": p} for p in family_prompts)

    return prompts


def _slugify(value: str, max_len: int = 80) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean[:max_len] or "prompt"


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    aliases = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
    key = aliases.get(dtype_str, dtype_str)
    if not hasattr(torch, key):
        raise ValueError(f"Unknown torch dtype: {dtype_str}")
    return getattr(torch, key)


def _ensure_circuit_tracer(auto_install: bool = True) -> None:
    try:
        import circuit_tracer  # noqa: F401
        return
    except ModuleNotFoundError:
        if not auto_install:
            raise RuntimeError(
                "Missing dependency `circuit_tracer`. Run `bash setup_colab.sh` "
                "or install with `%pip install git+https://github.com/decoderesearch/circuit-tracer.git`."
            )

    console.print("[yellow]`circuit_tracer` not found. Attempting automatic install...[/yellow]")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/decoderesearch/circuit-tracer.git",
    ]
    subprocess.run(cmd, check=True)
    try:
        import circuit_tracer  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Auto-install attempted but `circuit_tracer` is still unavailable. "
            "In Colab, run `%pip install -q git+https://github.com/decoderesearch/circuit-tracer.git` "
            "in the same notebook kernel, then retry."
        ) from exc


def _init_replacement_model(cfg: GenerationConfig, device: str = "cuda"):
    _ensure_circuit_tracer(auto_install=True)
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    dtype = _dtype_from_string(cfg.dtype)
    console.print(
        f"[cyan]Loading transcoders[/cyan]: set={cfg.transcoder_set}, dtype={dtype}, device={device}"
    )
    transcoders, transcoder_cfg = load_transcoder_from_hub(
        cfg.transcoder_set,
        dtype=dtype,
        lazy_encoder=cfg.lazy_encoder,
        lazy_decoder=cfg.lazy_decoder,
    )
    model_name = cfg.model or transcoder_cfg.get("model_name", "")
    if not model_name:
        raise ValueError("Model name missing. Set generation.model in config.")

    console.print(
        f"[cyan]Loading replacement model[/cyan]: model={model_name}, backend={cfg.backend}"
    )
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name,
        transcoders,
        backend=cfg.backend,
        dtype=dtype,
        device=torch.device(device) if device in {"cuda", "cpu"} else None,
    )
    return model


def _graph_counts_from_json(json_path: Path) -> tuple[int, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload.get("nodes", [])), len(payload.get("links", []))


def _generate_one_python_api(
    model: Any,
    prompt: str,
    family: str,
    slug: str,
    output_dir: Path,
    cfg: GenerationConfig,
) -> dict[str, Any]:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files

    start = time.time()
    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=cfg.max_n_logits,
        desired_logit_prob=cfg.desired_logit_prob,
        batch_size=cfg.batch_size,
        offload=cfg.offload,
        max_feature_nodes=cfg.max_feature_nodes,
        verbose=False,
    )

    pt_dir = output_dir / "pt"
    json_dir = output_dir / "json"
    meta_dir = output_dir / "metadata"
    pt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    pt_path = pt_dir / f"{slug}.pt"
    graph.to_pt(str(pt_path))

    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=str(json_dir),
        node_threshold=cfg.node_threshold,
        edge_threshold=cfg.edge_threshold,
    )
    json_path = json_dir / f"{slug}.json"
    n_nodes, n_edges = _graph_counts_from_json(json_path)

    elapsed = time.time() - start
    meta = {
        "slug": slug,
        "family": family,
        "prompt": prompt,
        "pt_path": str(pt_path),
        "json_path": str(json_path),
        "nodes": n_nodes,
        "edges": n_edges,
        "seconds": elapsed,
        "status": "ok",
    }
    with open(meta_dir / f"{slug}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _generate_one_cli_fallback(
    prompt: str,
    family: str,
    slug: str,
    output_dir: Path,
    cfg: GenerationConfig,
    device: str,
) -> dict[str, Any]:
    start = time.time()
    pt_dir = output_dir / "pt"
    json_dir = output_dir / "json"
    meta_dir = output_dir / "metadata"
    pt_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    pt_path = pt_dir / f"{slug}.pt"

    cmd = [
        "python",
        "-m",
        "circuit_tracer",
        "attribute",
        "--model",
        cfg.model,
        "--transcoder_set",
        cfg.transcoder_set,
        "--prompt",
        prompt,
        "--graph_output_path",
        str(pt_path),
        "--slug",
        slug,
        "--graph_file_dir",
        str(json_dir),
        "--offload",
        cfg.offload or "cpu",
        "--dtype",
        cfg.dtype,
        "--max_feature_nodes",
        str(cfg.max_feature_nodes),
        "--node_threshold",
        str(cfg.node_threshold),
        "--edge_threshold",
        str(cfg.edge_threshold),
        "--backend",
        cfg.backend,
    ]
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if run.returncode != 0:
        raise RuntimeError(run.stderr.strip() or run.stdout.strip() or "CLI generation failed")

    json_path = json_dir / f"{slug}.json"
    n_nodes, n_edges = _graph_counts_from_json(json_path)
    elapsed = time.time() - start
    meta = {
        "slug": slug,
        "family": family,
        "prompt": prompt,
        "pt_path": str(pt_path),
        "json_path": str(json_path),
        "nodes": n_nodes,
        "edges": n_edges,
        "seconds": elapsed,
        "status": "ok",
        "mode": "cli_fallback",
    }
    with open(meta_dir / f"{slug}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def generate_graphs(
    prompts: list[dict[str, str]],
    output_dir: str | Path,
    cfg: GenerationConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    _ensure_circuit_tracer(auto_install=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log_path = output_dir / "failed_prompts.jsonl"

    console.print(f"[bold]Generating {len(prompts)} attribution graphs[/bold]")
    start_all = time.time()

    model = None
    python_mode_ok = True
    try:
        model = _init_replacement_model(cfg, device=device)
    except Exception as exc:
        python_mode_ok = False
        console.print(
            f"[yellow]Python API init failed ({exc}). Falling back to CLI per prompt.[/yellow]"
        )

    stats: list[dict[str, Any]] = []
    failures = 0

    use_rich = True
    progress = None
    task = None
    try:
        progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        progress.start()
        task = progress.add_task("Generating graphs...", total=len(prompts))
    except Exception:
        use_rich = False

    iterator = prompts if use_rich else tqdm(prompts, desc="Generating graphs", unit="graph")

    for i, item in enumerate(iterator, start=1):
        prompt = item["prompt"]
        family = item["family"]
        slug = f"{family}_{i:04d}_{_slugify(prompt)}"
        try:
            if python_mode_ok and model is not None:
                meta = _generate_one_python_api(model, prompt, family, slug, output_dir, cfg)
            else:
                meta = _generate_one_cli_fallback(prompt, family, slug, output_dir, cfg, device=device)
            stats.append(meta)
            eta_hint = ""
            if use_rich and progress is not None:
                progress.update(
                    task,  # type: ignore[arg-type]
                    advance=1,
                    description=(
                        f"[cyan][Family: {family}] {i}/{len(prompts)} "
                        f"| Nodes: {meta['nodes']} | Edges: {meta['edges']} | Time: {meta['seconds']:.1f}s{eta_hint}"
                    ),
                )
            else:
                console.print(
                    f"[Family: {family}] Graph {i}/{len(prompts)} | Nodes: {meta['nodes']} "
                    f"| Edges: {meta['edges']} | Time: {meta['seconds']:.1f}s"
                )
        except Exception as exc:
            failures += 1
            with open(failed_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"family": family, "prompt": prompt, "error": str(exc)}) + "\n")
            console.print(f"[red]Failed[/red] {family} | {prompt[:70]}... -> {exc}")
            if use_rich and progress is not None:
                progress.update(task, advance=1)  # type: ignore[arg-type]
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if use_rich and progress is not None:
        progress.stop()

    total_time = time.time() - start_all
    avg_nodes = sum(s["nodes"] for s in stats) / max(1, len(stats))
    avg_edges = sum(s["edges"] for s in stats) / max(1, len(stats))

    summary = {
        "total_requested": len(prompts),
        "total_generated": len(stats),
        "failed": failures,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges,
        "total_seconds": total_time,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(
        "[bold green]Generation complete[/bold green] "
        f"| generated={summary['total_generated']} failed={summary['failed']} "
        f"| avg_nodes={summary['avg_nodes']:.1f} avg_edges={summary['avg_edges']:.1f} "
        f"| total_time={summary['total_seconds']:.1f}s"
    )
    return summary
