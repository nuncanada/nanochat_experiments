#!/bin/bash

# Recursive Reasoning CPU "Better" Test Run
# Aiming for ~20-30 minutes on CPU to see more learning progress.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# 0) Tokenizer (using 4 shards for better data variety)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer..."
    python -m nanochat.dataset -n 4
    python -m scripts.tok_train --max-chars=10000000
fi

# 1) Train a 4-layer base model on CPU
echo "Step 1: Training base model (Depth 4) on CPU..."
python -m scripts.base_train \
    --device-type=cpu \
    --depth=4 \
    --aspect-ratio=16 \
    --max-seq-len=128 \
    --device-batch-size=4 \
    --total-batch-size=512 \
    --num-iterations=100 \
    --eval-tokens=2048 \
    --eval-every=50 \
    --run="dummy" \
    --model-tag="better_base" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=100

# 2) Transform it to RecursiveGPT
echo "Step 2: Transforming to RecursiveGPT..."
python -m scripts.transform_to_recursive \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/better_base" \
    --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/better_recursive" \
    --step 100

# 3) Retrain Recursive Model on CPU
# Compressing 4 layers into 2 recursive steps.
echo "Step 3: Retraining Recursive model (4 layers -> 2 steps) on CPU..."
python -m scripts.recursive_train \
    --device-type=cpu \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/better_recursive" \
    --resume-from-step 100 \
    --n-steps 2 \
    --num-iterations=200 \
    --device-batch-size=4 \
    --eval-tokens=2048 \
    --eval-every=50 \
    --run="dummy" \
    --dependency-lr=0.05
