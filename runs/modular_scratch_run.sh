#!/bin/bash

# Training MGR-ALRT (Modular Grouped Recursive) from scratch
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
ITER=1000
BATCH=16
GROUPS=8

echo "=== Training MGR-ALRT Model (Modular Grouped) from Scratch (1000 iterations) ==="
# This uses 8 parallel branches of dim C/8 inside each of the 8 dynamic steps.
python -m scripts.modular_alrt_train \
    --device-type=cpu \
    --depth=$DEPTH \
    --aspect-ratio=$ASPECT \
    --max-seq-len=$SEQ \
    --n-groups $GROUPS \
    --n-steps 8 \
    --top-fixed 2 \
    --bottom-fixed 2 \
    --ponder-weight 0.01 \
    --device-batch-size=$BATCH \
    --num-iterations=$ITER \
    --eval-tokens=4096 \
    --eval-every=100 \
    --model-tag="scratch_mgr_alrt_16l_1000"

echo ""
echo "=== MGR-ALRT Training Complete ==="
