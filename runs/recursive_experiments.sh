#!/bin/bash

# Recursive Reasoning Compression Experiments
# Goal: Compress 16 layers into fewer steps while monitoring recovery.

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# --- STEP 1: Train a stronger 16-layer base model ---
# We'll use 200 iterations to get a more stable baseline.
echo "=== Phase 1: Training Base 16L Model (200 iterations) ==="
python -m scripts.base_train \
    --device-type=cpu \
    --depth=16 \
    --aspect-ratio=8 \
    --max-seq-len=64 \
    --device-batch-size=4 \
    --total-batch-size=512 \
    --num-iterations=200 \
    --eval-tokens=1024 \
    --eval-every=50 \
    --run="dummy" \
    --model-tag="exp_16l_base" \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=200

BASE_CHECKPOINT="$NANOCHAT_BASE_DIR/base_checkpoints/exp_16l_base"

# Helper function for experiments
run_exp() {
    local n_steps=$1
    local mid_steps=$((n_steps - 4))
    local tag="exp_16l_to_${n_steps}s"
    
    echo ""
    echo "=== Experiment: 16 Layers -> $n_steps Total Steps ($mid_steps Middle Steps) ==="
    
    # Transform
    python -m scripts.transform_to_recursive \
        --checkpoint-dir "$BASE_CHECKPOINT" \
        --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --step 200
        
    # Retrain (100 iterations with high dependency LR to speed up recovery)
    python -m scripts.recursive_train \
        --device-type=cpu \
        --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --resume-from-step 200 \
        --n-steps $n_steps \
        --top-fixed 2 \
        --bottom-fixed 2 \
        --num-iterations=300 \
        --device-batch-size=4 \
        --eval-tokens=1024 \
        --eval-every=25 \
        --run="dummy" \
        --dependency-lr=0.1
}

# --- STEP 2: Run Experiments ---

# A: Moderate Compression (16L -> 10 steps: 2 + 6 + 2)
run_exp 10

# B: High Compression (16L -> 6 steps: 2 + 2 + 2)
run_exp 6

# C: Extreme Compression (16L -> 5 steps: 2 + 1 + 2)
run_exp 5

echo ""
echo "=== All Experiments Complete ==="
