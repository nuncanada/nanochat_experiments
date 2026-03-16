#!/bin/bash

# Recursive Reasoning CPU 16-Layer Test Run
# Config: 16 layers, top 2 and bottom 2 fixed.
# Compression: 16 layers -> 8 steps total.
# Steps breakdown:
# Step 0: Block 0 (Fixed)
# Step 1: Block 1 (Fixed)
# Steps 2-5: Blocks 2-13 (Trainable middle)
# Step 6: Block 14 (Fixed)
# Step 7: Block 15 (Fixed)

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
    --num-iterations=50 \
    --eval-tokens=512 \
    --eval-every=25 \
    --run="dummy" \
    --model-tag="16l_base" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=50

# 2) Transform it to RecursiveGPT
echo "Step 2: Transforming to RecursiveGPT..."
python -m scripts.transform_to_recursive \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/16l_base" \
    --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/16l_recursive" \
    --step 50

# 3) Retrain Recursive Model on CPU
echo "Step 3: Retraining Recursive model (16L -> 8 steps, top/bottom 2 fixed)..."
python -m scripts.recursive_train \
    --device-type=cpu \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/16l_recursive" \
    --resume-from-step 50 \
    --n-steps 8 \
    --top-fixed 2 \
    --bottom-fixed 2 \
    --num-iterations=150 \
    --device-batch-size=2 \
    --eval-tokens=512 \
    --eval-every=25 \
    --run="dummy" \
    --dependency-lr=0.05
