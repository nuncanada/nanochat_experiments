#!/bin/bash

# Research test run based on README.md Research section, adapted for CPU.
# This runs a small d12 model (GPT-1 sized) for a quick experiment on CPU.
# NOTE: Training on CPU is slow, so we further reduce parameters and iterations 
# for a "test" that actually finishes in a reasonable time.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# Run the base d12 model training but with CPU-friendly settings.
# We use a smaller sequence length and batch size to avoid OOM and speed up execution.
python -m scripts.base_train \
    --device-type=cpu \
    --depth=12 \
    --aspect-ratio=32 \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=1024 \
    --num-iterations=100 \
    --run="d12_research_cpu" \
    --model-tag="d12_research_cpu" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1
