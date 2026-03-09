#!/bin/bash
##  Asking for 1 node, and 6 cores - the next line asks for 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -o outLog
#SBATCH -e errLog

for seed in 0 1 2 3 4
do
    python -m train_transfer \
        --results_dir "checkpoints/lr_exp_4" \
        --epochs 250 \
        --learning_rate 1e-4 \
        --seed ${seed}

    python -m train_transfer \
        --results_dir "checkpoints/lr_exp_5" \
        --epochs 250 \
        --learning_rate 1e-5 \
        --seed ${seed}
done