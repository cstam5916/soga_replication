import argparse
import os

import numpy as np
import torch
from torch import nn

from models import GCN
from DomainData import DomainData

@torch.no_grad()
def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    if pred.numel() == 0:
        return 0.0
    num_classes = int(max(pred.max().item(), y.max().item())) + 1
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1s.append(f1)
    return sum(f1s) / len(f1s)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acm_data = DomainData('data/acm', 'acm')[0].to(device)
    dblp_data = DomainData('data/dblp', 'dblp')[0].to(device)
    datasets = [acm_data, dblp_data]

    in_channels = acm_data.x.size(-1)
    out_channels = int(acm_data.y.max().item()) + 1

    model = GCN(
        in_channels=in_channels,
        hidden_channels=[256, 256, 128],
        out_channels=out_channels,
    ).to(device)
    acm_state_dict = torch.load(os.path.join("checkpoints/acm_gcn_exact/best_model.pt"))['model_state_dict']
    dblp_state_dict = torch.load(os.path.join("checkpoints/dblp_gcn_exact/best_model.pt"))['model_state_dict']
    state_dicts = acm_state_dict, dblp_state_dict


    for model_name, model_state_dict in zip(['acm', 'dblp'], state_dicts):
        model.load_state_dict(model_state_dict)
        for data_name, data in zip(['acm', 'dblp'], datasets):
            model.eval()
            out = model(data)
            pretrained_acc = macro_f1(out, data.y)
            print(f'Accuracy of transfer {model_name} -> {data_name}: {pretrained_acc}')
        
if __name__ == '__main__':
    main()