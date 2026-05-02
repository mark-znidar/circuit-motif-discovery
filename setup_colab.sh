#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Circuit Motif Discovery — Environment Setup"
echo "============================================"

# Detect PyTorch and CUDA versions from existing Colab install
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+v.replace('.','')) if v else print('cpu')")
echo "[1/6] Detected torch=${TORCH_VERSION} cuda=${CUDA_TAG}"

# Install PyTorch Geometric + extensions
echo "[2/6] Installing PyTorch Geometric..."
python -m pip install -q torch-geometric
python -m pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html" 2>/dev/null || \
    echo "  Warning: Some PyG extensions failed — this is often OK on newer CUDA"

# Install circuit-tracer
echo "[3/6] Installing circuit-tracer..."
python -m pip install -q circuit-tracer 2>/dev/null || \
    python -m pip install -q git+https://github.com/decoderesearch/circuit-tracer.git

# Install remaining dependencies
echo "[4/6] Installing project dependencies..."
python -m pip install -q hdbscan umap-learn scikit-learn matplotlib seaborn \
    pyyaml tqdm requests rich

# Install local package
echo "[5/6] Installing local project..."
python -m pip install -q -e .

# Smoke test
echo "[6/6] Running smoke test..."
python -c "
import torch, torch_geometric
print(f'  torch {torch.__version__}')
print(f'  pyg   {torch_geometric.__version__}')
print(f'  cuda  {torch.cuda.is_available()}')
try:
    import circuit_tracer
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.attribution.attribute import attribute
    print('  circuit-tracer OK')
except Exception as exc:
    raise RuntimeError(f'circuit-tracer import failed ({exc})')
from torch_geometric.nn import GINEConv
print('  GINEConv OK')
print('All checks passed!')
"
