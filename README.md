# Replication Project: Source Free Unsupervised Graph Domain Adaptation
This repository contains a partial replication of the paper "Source Free Unsupervised Graph Domain Adaptation" by Haitao Mao, Lun Du, Yujia Zheng, Qiang Fu, Zelin Li, and Xu Chen, published in WSDM 2024. Their paper can be found be found [here](https://dl.acm.org/doi/abs/10.1145/3616855.3635802) and their Github repository [here](https://github.com/HaitaoMao/SOGA/tree/main).

This project is done for credit as part of the graduate course "CMPSC 292F: Information Theory for Trustworthy Machine Learning," offered at UC Santa Barbara in Winter 2026, taught by Professor Yuheng Bu.

# Replication Instructions
## Environment Setup
For ease of managing dependencies, we recommend that you first install the following dependencies with manually-specified wheels (torch & cuda versions changed as necessary)
``` 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```
before installing the remaining requirements with
``` 
pip install -r requirements/requirements.txt --override requirements/override.txt
```
If any additional problems with dependencies arise, we recommend using the [uv package manager](https://docs.astral.sh/uv/) for environment setup.

## Dataset Acquisition
The datasets used in this projects are the preprocessed versions of the ACMv8 and DBLPv9 datasets from the paper [Unsupervised Domain Adaptive Graph Convolutional Networks](https://dl.acm.org/doi/pdf/10.1145/3366423.3380219) ([Github](https://github.com/TrustAGI-Lab/UDAGCN/tree/master)) and can be accessed at [this Google Drive link](https://drive.google.com/file/d/1DzQ3QN9yjQxU4vtYkXyCiJKFw7oCCPSM/view?usp=sharing). 


## Running Scripts
A .sh file with the following scripts run in sequence is provided in `replicate.sh`. This shell file runs a sequence of three Python scripts, corresponding to
1. Training models on each source dataset (ACM & DBLP)
2. Fine-tuning these models on the other dataset, using the full SOGA loss function and two ablations (IMOnly & SCOnly)
3. Generating plots for the train/validation F1 scores for each adaptation direction.
After running the scripts, models and training logs will be saved in the `checkpoints` directory. Plots will be saved as `.svg` files in the `results` directory alongisde a .txt file with the best F1-scores on each task. Precomputed results are also given.

## Technical discrepancies in this implementation
1. The original publication was not completely exhaustive in its description of the graph convolutional network used. For example, it is unclear whether the architecture in their implementation included layers like batch/layernorm, or whether linear or convolutional layers were used for projection between the input/output dimensions and the hidden dimension. Because performance is sensitive with respect to architectural choices, we believe this is the primary explanation for differences in zero-shot performance between this implementation and the paper's.
    - Note (for instructors & classmates): An additional hidden layer was added between the in-class presentation for CMPSC 292F on 3/13/2026 and the final version of this code. This has led to plots that more closely resemble those presented in the SOGA paper, but differ from those seen in that presentation.
2. For computational efficiency and ease of implementation, we use the (Role2Vec)[https://ieeexplore.ieee.org/iel7/69/4358933/09132694.pdf] algorithm to generate role-based structural embeddings for nodes rather than (Struc2Vec)[https://dl.acm.org/doi/pdf/10.1145/3097983.3098061], which was used in the SOGA publication. Because the SOGA pipeline only uses these embeddings to find top-K neighbors in embedding space, we hypothesize that this change would have a minor impact on performance. However, it may explain the discrepancy in stability of structural consistency training, particularly on the DBLP to ACM task.