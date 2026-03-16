#!/bin/bash

# Recursive Reasoning CPU Dynamic (Router) Test Run
# Config: 16 layers, top 2 and bottom 2 fixed.
# Compression: 16 layers -> 8 steps total with Dynamic Halting.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# 1) Train a 16-layer tiny base model on CPU
echo "Step 1: Training 16-layer base model on CPU..."
python -m scripts.base_train \
    --device-type=cpu \
    --depth=16 \
    --aspect-ratio=8 \
    --max-seq-len=64 \
    --device-batch-size=2 \
    --total-batch-size=128 \
    --num-iterations=20 \
    --eval-tokens=512 \
    --eval-every=10 \
    --run="dummy" \
    --model-tag="dynamic_16l_base" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=20

# 2) Transform it to RecursiveGPT
echo "Step 2: Transforming to RecursiveGPT..."
python -m scripts.transform_to_recursive \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/dynamic_16l_base" \
    --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/dynamic_16l_recursive" \
    --step 20

# 3) Retrain Dynamic Recursive Model on CPU
echo "Step 3: Retraining Dynamic Recursive model (16L -> 8 steps, with Router)..."
python -m scripts.recursive_train \
    --device-type=cpu \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/dynamic_16l_recursive" \
    --resume-from-step 20 \
    --n-steps 8 \
    --top-fixed 2 \
    --bottom-fixed 2 \
    --num-iterations=50 \
    --device-batch-size=2 \
    --eval-tokens=512 \
    --eval-every=10 \
    --run="dummy" \
    --dependency-lr=0.05
