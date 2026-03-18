# Replication Project: Source Free Unsupervised Graph Domain Adaptation
This repository contains a partial replication of the paper "Source Free Unsupervised Graph Domain Adaptation" by Haitao Mao, Lun Du, Yujia Zheng, Qiang Fu, Zelin Li, and Xu Chen, published in WSDM 2024. Their paper can be found be found [here](https://dl.acm.org/doi/abs/10.1145/3616855.3635802) and their Github repository [here](https://github.com/HaitaoMao/SOGA/tree/main).

This project is done for credit as part of the graduate course "CMPSC 292F: Information Theory for Trustworthy Machine Learning," offered at UC Santa Barbara in Winter 2026, taught by Professor Yuheng Bu.

# Replication Instructions
## Environment Setup
For ease of managing dependencies, we recommend that you first install the following dependencies with manually-specified wheels (torch & cuda versions changed as necessary)
``` pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html ```
before installing the remaining requirements with
``` pip install -r requirements.txt ```
If any additional problems with dependencies arise, we recommend using the [uv package manager](https://docs.astral.sh/uv/) for environment setup.

## Dataset Acquisition

## Running Scripts
A .sh file with the following scripts run in sequence is provided in `replicate.sh`. This script uses the following protocol:

First, train the source models with
``` python -m train --root "data/acm" --results_dir "checkpoints/acm_gcn" ```
``` python -m train --root "data/dblp" --results_dir "checkpoints/dblp_gcn" ```
changing the source dataset and save path freely. Second, fine-tune the model on the domain adaptation objective with
``` python train_transfer.py --root "./data/dblp" --source_model "./checkpoints/acm_gcn" --results_dir "checkpoints/dblp_to_acm_SOGA" --mode "SOGA" ```
``` python train_transfer.py --root "./data/acm" --source_model "./checkpoints/dblp_gcn" --results_dir "checkpoints/acm_to_dblp_SOGA" --mode "SOGA" ```
