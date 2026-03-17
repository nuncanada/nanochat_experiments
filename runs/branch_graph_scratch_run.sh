#!/bin/bash

# Training BG-ALRT (Vectorized Branch Graph) from scratch
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

echo "=== Training BG-ALRT Model (Vectorized Branch Graph) from Scratch (1000 iterations) ==="
# Optimized vectorized model
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
    --model-tag="scratch_bg_alrt_vectorized"

echo ""
echo "=== Final Comparison Table ==="
echo "Model | Architecture | Final BPB | Steps (Avg)"
echo "--------------------------------------------------"
# We retrieve base BPB from previous 500 iter run (which was ~2.16)
# and compare with this 1000 iter BG-ALRT run.
echo "Base | Sequential | 2.16 | 16.0"
# Final BPB will be printed by the train script above.
