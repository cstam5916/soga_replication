# Training source models
python -m train --root "data/acm" --results_dir "checkpoints/acm_gcn"
python -m train --root "data/dblp" --results_dir "checkpoints/dblp_gcn"

# Training on domain adaptation objectives
MODES=("SCOnly" "IMOnly" "SOGA")

for MODE in "${MODES[@]}"; do
  python train_transfer.py \
    --root "./data/acm" \
    --source_model "./checkpoints/dblp_gcn" \
    --results_dir "./checkpoints/dblp_to_acm_${MODE}" \
    --mode "${MODE}"
done

for MODE in "${MODES[@]}"; do
  python train_transfer.py \
    --root "./data/dblp" \
    --source_model "./checkpoints/acm_gcn" \
    --results_dir "./checkpoints/acm_to_dblp_${MODE}" \
    --mode "${MODE}"
done

# Generating visualizations
python make_plots.py