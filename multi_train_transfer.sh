#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

EMBED_VARIANTS=("SOGA" "SCOnly")
NONEMBED_VARIANTS=("IMOnly")
EMBED_MODES=("Role2Vec" "Struc2Vec")

TRANSFERS=(
    "data/dblp checkpoints/source/acm"
    "data/acm  checkpoints/source/dblp"
)

# ---------------------------------------------------------------------------

for TRANSFER in "${TRANSFERS[@]}"; do
    read -r ROOT SOURCE_MODEL <<< "$TRANSFER"
    for VARIANT in "${EMBED_VARIANTS[@]}"; do
        for EMBED_MODE in "${EMBED_MODES[@]}"; do
            MODE="SSFUG_${VARIANT}_${EMBED_MODE}"
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
    for VARIANT in "${NONEMBED_VARIANTS[@]}"; do
        MODE="SSFUG_${VARIANT}"
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
