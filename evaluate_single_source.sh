#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_single_source
#SBATCH --output=logs/eval_single_source_%j.out
#SBATCH --error=logs/eval_single_source_%j.err

# ---------------------------------------------------------------------------
# evaluate_single_source.sh — ensemble evaluation of single-source transfer models
# Loops over Citationv1, ACMv9, DBLPv7 as targets, using models trained on the
# other two datasets (SSFUG_SOGA_Struc2Vec) as the ensemble sources.
# Submit with:  sbatch evaluate_single_source.sh
# ---------------------------------------------------------------------------

DATASETS=("Citationv1" "ACMv9" "DBLPv7")
MODEL_DIR="SSFUG_SOGA_Struc2Vec"

for TARGET in "${DATASETS[@]}"; do
    MODELS=()
    for SOURCE in "${DATASETS[@]}"; do
        if [ "$SOURCE" != "$TARGET" ]; then
            MODELS+=("checkpoints/SSFUG/${SOURCE}_to_${TARGET}/${MODEL_DIR}")
        fi
    done

    python -m scripts.evaluate_single_source \
        --root   "dataset/${TARGET}" \
        --models "${MODELS[@]}"
done