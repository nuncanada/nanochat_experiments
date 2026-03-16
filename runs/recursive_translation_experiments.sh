#!/bin/bash

# Recursive Reasoning with Translation Matrices (Layer Adapters)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# --- PHASE 1: Base Model (Re-using or training 500 iterations) ---
# We already have exp_16l_base from previous run, let's use it or re-train if missing.
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/base_16l_500" ]; then
    echo "Training Stronger Base Model (500 iterations)..."
    python -m scripts.base_train \
        --device-type=cpu \
        --depth=16 \
        --aspect-ratio=8 \
        --max-seq-len=64 \
        --device-batch-size=4 \
        --total-batch-size=512 \
        --num-iterations=500 \
        --eval-tokens=1024 \
        --eval-every=100 \
        --run="dummy" \
        --model-tag="base_16l_500" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=500
fi

BASE_CHECKPOINT="$NANOCHAT_BASE_DIR/base_checkpoints/base_16l_500"

# --- PHASE 2: Translation Matrix Experiments ---
run_trans_exp() {
    local n_steps=$1
    local ponder=$2
    local tag="trans_16l_s${n_steps}_p${ponder}"
    
    echo ""
    echo "=== Experiment (Translation): steps=$n_steps, ponder=$ponder ==="
    
    python -m scripts.transform_to_recursive \
        --checkpoint-dir "$BASE_CHECKPOINT" \
        --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --step 500
        
    # Retrain for 300 iterations (total 800) to allow translation matrices to converge
    python -m scripts.recursive_train \
        --device-type=cpu \
        --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --resume-from-step 500 \
        --n-steps $n_steps \
        --top-fixed 2 \
        --bottom-fixed 2 \
        --ponder-weight $ponder \
        --num-iterations=800 \
        --device-batch-size=4 \
        --eval-tokens=1024 \
        --eval-every=50 \
        --run="dummy" \
        --dependency-lr=0.05
}

# 1. Standard Compression with Translation (8 steps)
run_trans_exp 8 0.01

# 2. Higher Efficiency with Translation (6 steps)
run_trans_exp 6 0.01

# 3. High Ponder with Translation (8 steps, high ponder)
run_trans_exp 8 0.05

echo ""
echo "=== All Translation Experiments Complete ==="
