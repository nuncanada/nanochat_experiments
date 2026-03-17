#!/bin/bash

# Comparison: Base (Sequential) vs ALRT (Recursive) trained from scratch
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# Parameters
DEPTH=16
ASPECT=8
SEQ=64
ITER=500
BATCH=16

echo "=== Phase 1: Training Base Sequential Model from Scratch ==="
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/scratch_base_16l" ]; then
    python -m scripts.base_train \
        --device-type=cpu \
        --depth=$DEPTH \
        --aspect-ratio=$ASPECT \
        --max-seq-len=$SEQ \
        --device-batch-size=$BATCH \
        --total-batch-size=$((BATCH * 128)) \
        --num-iterations=$ITER \
        --eval-tokens=4096 \
        --eval-every=100 \
        --run="dummy" \
        --model-tag="scratch_base_16l" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=$ITER
else
    echo "Base model already exists, skipping."
fi

echo ""
echo "=== Phase 2: Training ALRT Model from Scratch ==="
# Note: ALRT uses 8 steps for 16 layers (dynamic)
python -m scripts.alrt_train \
    --device-type=cpu \
    --depth=$DEPTH \
    --aspect-ratio=$ASPECT \
    --max-seq-len=$SEQ \
    --n-steps 8 \
    --top-fixed 2 \
    --bottom-fixed 2 \
    --ponder-weight 0.01 \
    --device-batch-size=$BATCH \
    --num-iterations=$ITER \
    --eval-tokens=4096 \
    --eval-every=100 \
    --model-tag="scratch_alrt_16l"

echo ""
echo "=== Comparison Complete ==="
