#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# local_train_transfer.sh — run transfer training with train_transfer.py
# Edit the variables below, then run:  bash local_train_transfer.sh
# ---------------------------------------------------------------------------

ROOT="data/dblp"
SOURCE_MODEL="checkpoints/source/acm"
MODE="SSFUG_SCOnly_Struc2Vec"   # ALGOTYPE_VARIANT_EMBEDMODE
EPOCHS=100
LR=1e-3
SEED=0
FREEZE_AFTER=-1              # -1 = no freezing

# ---------------------------------------------------------------------------

python -m scripts.train_transfer \
    --root          "$ROOT" \
    --source_model  "$SOURCE_MODEL" \
    --mode          "$MODE" \
    --epochs        "$EPOCHS" \
    --learning_rate "$LR" \
    --seed          "$SEED" \
    --freeze_after  "$FREEZE_AFTER"
