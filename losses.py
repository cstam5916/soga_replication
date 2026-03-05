import torch
from torch import nn
import torch.nn.functional as F

class SOGALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, train_mask, edge_index):
        return self._conditional_entropy_loss(logits)
        
    def _conditional_entropy_loss(logits):
            log_q_cond = F.log_softmax(logits, dim=1)
            q_cond = log_q_cond.exp()
            ent = -(q_cond * log_q_cond).sum(dim=1).mean()
            return ent
    
    def _marginal_entropy_loss():
            return None
    
    def _structural_consistency_loss():
        return None