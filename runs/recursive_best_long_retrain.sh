#!/bin/bash

# Long Retraining of the best Recursive configuration
# Config: 8 Steps, Top 2 / Bottom 2 Fixed, Translation Matrices, Ponder 0.01
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
mkdir -p $NANOCHAT_BASE_DIR

# Setup
source .venv/bin/activate

BASE_CHECKPOINT="$NANOCHAT_BASE_DIR/base_checkpoints/base_16l_500"

# Best Configuration Experiment
TAG="best_16l_s8_p0.01_long"
echo "=== Retraining Best Configuration: $TAG ==="

# 1. Transform from the 500-step base model
python -m scripts.transform_to_recursive \
    --checkpoint-dir "$BASE_CHECKPOINT" \
    --output-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$TAG" \
    --step 500

# 2. Retrain for 500 iterations (from 500 to 1000)
python -m scripts.recursive_train \
    --device-type=cpu \
    --checkpoint-dir "$NANOCHAT_BASE_DIR/recursive_checkpoints/$TAG" \
    --resume-from-step 500 \
    --n-steps 8 \
    --top-fixed 2 \
    --bottom-fixed 2 \
    --ponder-weight 0.01 \
    --num-iterations=1000 \
    --device-batch-size=4 \
    --eval-tokens=4096 \
    --eval-every=100 \
    --run="dummy" \
    --dependency-lr=0.02
