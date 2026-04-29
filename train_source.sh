#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# local_train_source.sh — train a source GCN checkpoint
# Edit the variables below, then run:  bash local_train_source.sh
# ---------------------------------------------------------------------------

ROOT="dataset/Citationv1"
EPOCHS=200
LR=1e-3
SEED=0

# ---------------------------------------------------------------------------

python -m scripts.train \
    --root          "$ROOT" \
    --epochs        "$EPOCHS" \
    --learning_rate "$LR" \
    --seed          "$SEED"
