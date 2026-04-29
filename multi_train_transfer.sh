#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=multi_train_transfer
#SBATCH --output=logs/multi_train_transfer_%j.out
#SBATCH --error=logs/multi_train_transfer_%j.err

# ---------------------------------------------------------------------------
# multi_train_transfer.sh — run all implemented algo/dataset combinations
# Requires source checkpoints at checkpoints/source/acm and checkpoints/source/dblp
# (train those first with train_source.sh)
# ---------------------------------------------------------------------------

EPOCHS=100
LR=1e-3
SEED=0
FREEZE_AFTER=-1

EMBED_MODES=("Struc2Vec")

TRANSFERS=(
    "dataset/DBLPv7      checkpoints/source/ACMv9"
    "dataset/Citationv1  checkpoints/source/ACMv9"
    "dataset/ACMv9       checkpoints/source/DBLPv7"
    "dataset/Citationv1  checkpoints/source/DBLPv7"
    "dataset/ACMv9       checkpoints/source/Citationv1"
    "dataset/DBLPv7      checkpoints/source/Citationv1"
)

# ---------------------------------------------------------------------------

for TRANSFER in "${TRANSFERS[@]}"; do
    read -r ROOT SOURCE_MODEL <<< "$TRANSFER"
    for EMBED_MODE in "${EMBED_MODES[@]}"; do
        MODE="SSFUG_SOGA_${EMBED_MODE}"
        echo "=== $MODE | source=$(basename $SOURCE_MODEL) -> target=$(basename $ROOT) ==="
        python -m scripts.train_transfer \
            --root                 "$ROOT" \
            --source_model         "$SOURCE_MODEL" \
            --mode                 "$MODE" \
            --epochs               "$EPOCHS" \
            --learning_rate        "$LR" \
            --seed                 "$SEED" \
            --freeze_after         "$FREEZE_AFTER"
    done
done
