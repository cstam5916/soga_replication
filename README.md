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
A .sh file with the following scripts run in sequence is provided in `replicate.sh`.
