# Training source models
python -m scripts.train --root "dataset/acm" --results_dir "checkpoints/acm_gcn"
python -m scripts.train --root "dataset/dblp" --results_dir "checkpoints/dblp_gcn"

# Training on domain adaptation objectives
MODES=("SCOnly" "IMOnly" "SOGA")

for MODE in "${MODES[@]}"; do
  python -m scripts.train_transfer \
    --root "./dataset/acm" \
    --source_model "./checkpoints/dblp_gcn" \
    --results_dir "./checkpoints/dblp_to_acm_${MODE}" \
    --mode "${MODE}"
done

for MODE in "${MODES[@]}"; do
  python -m scripts.train_transfer \
    --root "./dataset/dblp" \
    --source_model "./checkpoints/acm_gcn" \
    --results_dir "./checkpoints/acm_to_dblp_${MODE}" \
    --mode "${MODE}"
done

# Generating visualizations
python -m scripts.make_plots