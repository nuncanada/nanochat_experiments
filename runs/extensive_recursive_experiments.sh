#!/bin/bash

# Extensive Recursive Compression Experiments
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# --- PHASE 1: Stronger Base Model ---
echo "=== Training Stronger Base Model (500 iterations) ==="
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

BASE_CHECKPOINT="$NANOCHAT_BASE_DIR/base_checkpoints/base_16l_500"

# --- PHASE 2: Experimental Variations ---
run_extensive_exp() {
    local n_steps=$1
    local top_fixed=$2
    local bottom_fixed=$3
    local ponder=$4
    local tag="ext_16l_s${n_steps}_t${top_fixed}_b${bottom_fixed}_p${ponder}"
    
    echo ""
    echo "=== Experiment: steps=$n_steps, top=$top_fixed, bot=$bottom_fixed, ponder=$ponder ==="
    
    python -m scripts.transform_to_recursive \
        --checkpoint-dir "$BASE_CHECKPOINT" \
        --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --step 500
        
    python -m scripts.recursive_train \
        --device-type=cpu \
        --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$tag" \
        --resume-from-step 500 \
        --n-steps $n_steps \
        --top-fixed $top_fixed \
        --bottom-fixed $bottom_fixed \
        --ponder-weight $ponder \
        --num-iterations=700 \
        --device-batch-size=4 \
        --eval-tokens=1024 \
        --eval-every=50 \
        --run="dummy" \
        --dependency-lr=0.05
}

# Experimental Matrix
# 1. Conservative (12 steps, 2/2 fixed)
run_extensive_exp 12 2 2 0.01

# 2. Standard (8 steps, 2/2 fixed)
run_extensive_exp 8 2 2 0.01

# 3. High Efficiency Bias (8 steps, 2/2 fixed, high ponder)
run_extensive_exp 8 2 2 0.05

# 4. Aggressive (6 steps, 2/2 fixed)
run_extensive_exp 6 2 2 0.01

# 5. Reduced Anchor (8 steps, 1/1 fixed)
run_extensive_exp 8 1 1 0.01

echo ""
echo "=== All Extensive Experiments Complete ==="
