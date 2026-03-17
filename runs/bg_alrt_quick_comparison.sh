#!/bin/bash

# Fast estimation of BG-ALRT performance
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

echo "=== Fast Estimation of BG-ALRT Model (Branch Graph) ==="
# Running only 50 iterations to see the BPB improvement trend.
python -m scripts.modular_alrt_train \
    --device-type=cpu \
    --depth=16 \
    --aspect-ratio=8 \
    --max-seq-len=64 \
    --n-groups 8 \
    --n-steps 8 \
    --ponder-weight 0.01 \
    --device-batch-size=16 \
    --num-iterations=50 \
    --eval-tokens=4096 \
    --eval-every=25 \
    --model-tag="fast_bg_alrt"

echo ""
echo "=== Estimation Complete ==="
