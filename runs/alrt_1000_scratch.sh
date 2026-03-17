#!/bin/bash

# ALRT Scratch Training for 1000 iterations (Wall-clock comparison)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

# Parameters (Identical to comparison run but 1000 iterations)
DEPTH=16
ASPECT=8
SEQ=64
ITER=1000
BATCH=16

echo "=== Training ALRT Model from Scratch (1000 iterations) ==="
# This should take approximately the same wall-clock time as 500 Base iterations.
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
    --model-tag="scratch_alrt_16l_1000"

echo ""
echo "=== ALRT 1000 Iteration Training Complete ==="
