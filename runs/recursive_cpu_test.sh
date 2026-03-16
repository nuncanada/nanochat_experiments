#!/bin/bash

# Recursive Reasoning CPU Test Run (Minimal for quick validation)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled # Disable wandb prompts
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# 0) Train tokenizer if not present
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer..."
    python -m nanochat.dataset -n 1
    python -m scripts.tok_train --max-chars=1000000
fi

# 1) Train a tiny base model on CPU
echo "Step 1: Training tiny base model on CPU..."
python -m scripts.base_train \
    --device-type=cpu \
    --depth=2 \
    --aspect-ratio=8 \
    --max-seq-len=64 \
    --device-batch-size=2 \
    --total-batch-size=128 \
    --num-iterations=10 \
    --eval-tokens=128 \
    --eval-every=5 \
    --run="dummy" \
    --model-tag="tiny_base" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=10

# 2) Transform it to RecursiveGPT
echo "Step 2: Transforming to RecursiveGPT..."
python -m scripts.transform_to_recursive \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/tiny_base" \
    --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/tiny_recursive" \
    --step 10

# 3) Retrain Recursive Model on CPU
echo "Step 3: Retraining Recursive model on CPU..."
python -m scripts.recursive_train \
    --device-type=cpu \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/tiny_recursive" \
    --resume-from-step 10 \
    --n-steps 1 \
    --num-iterations=20 \
    --device-batch-size=2 \
    --eval-tokens=128 \
    --eval-every=5 \
    --run="dummy"
