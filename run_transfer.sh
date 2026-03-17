#!/bin/bash
##  Asking for 1 node, and 6 cores - the next line asks for 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -o outLog
#SBATCH -e errLog

python train_transfer.py --root "./data/acm" --source_model "./checkpoints/dblp_gcn_exact" --results_dir "checkpoints/dblp_to_acm_SOGA" --mode "SOGA"
python train_transfer.py --root "./data/acm" --source_model "./checkpoints/dblp_gcn_exact" --results_dir "checkpoints/dblp_to_acm_IMOnly" --mode "IMOnly"
python train_transfer.py --root "./data/acm" --source_model "./checkpoints/dblp_gcn_exact" --results_dir "checkpoints/dblp_to_acm_SCOnly" --mode "SCOnly"