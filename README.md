# Circuit Motif Discovery

Graph contrastive learning on LLM attribution graphs to discover recurring circuit motifs.

This MVP extracts attribution graphs from Gemma-2-2B with `circuit-tracer`, converts them to PyTorch Geometric objects, trains a GraphCL encoder, and evaluates whether learned graph embeddings recover prompt-family structure better than hand-crafted baselines. The output is a set of static figures and metrics intended for rapid research validation.

## Quick Start

```bash
cd circuit-motif-discovery
bash setup_colab.sh
python scripts/run_full_pipeline.py --quick
```

## Running on Google Colab Pro

1. Open Google Colab (`colab.research.google.com`)
2. Set runtime: Runtime -> Change runtime type -> GPU (A100 preferred, T4 works too)
3. Put the project in Colab (either method works):
   - **Google Drive (recommended):** upload `circuit-motif-discovery/` (or `circuit-motif-discovery.zip`) to `MyDrive`
   - **Direct upload:** upload the folder/zip to Colab's `/content` from the Files panel
4. Run setup from notebook Cell 1 (it auto-detects Drive vs `/content` paths), or run manually:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/circuit-motif-discovery
   !bash setup_colab.sh
   ```
5. Quick smoke test (~10 min):
   ```python
   !python scripts/run_full_pipeline.py --quick
   ```
6. Full run (~2-4 hours):
   ```python
   !python scripts/run_full_pipeline.py
   ```
7. View results:
   ```python
   from IPython.display import Image, display
   display(Image('results/demo1_umap.png'))
   display(Image('results/demo4_metrics.png'))
   ```

### Colab Walkthrough Screenshots

- `[Placeholder]` Runtime GPU selection screenshot
- `[Placeholder]` Setup cell output screenshot
- `[Placeholder]` Final results panel screenshot

## Project Structure

- `prompts/prompt_corpus.json`: 5 prompt families (250 prompts total)
- `src/graph_generation.py`: attribution graph generation via `circuit-tracer`
- `src/graph_converter.py`: JSON/PT to PyG `Data` conversion
- `src/augmentations.py`: circuit-aware graph augmentations
- `src/model.py`: GINE-based encoder + projection head
- `src/contrastive.py`: GraphCL training loop + NT-Xent loss
- `src/baselines.py`: non-GNN representation baselines
- `src/evaluation.py`: metrics, retrieval analysis, and static figures
- `scripts/`: end-to-end CLI pipeline
- `notebooks/colab_full_pipeline.ipynb`: single Colab-run notebook

## Expected Results

After a full run, look for:

- `results/demo1_umap.png`: clearer family separation for GNN embeddings
- `results/demo2_retrieval.png`: nearest-neighbor retrievals with family coherence
- `results/demo3_cluster_motifs.png`: cluster-wise motif summaries
- `results/demo4_metrics.png`: metric comparison across methods
- `results/metrics_summary.json`: machine-readable metrics

## Citation

If you use this project, cite:

```text
[Citation placeholder]
```

## License

MIT
