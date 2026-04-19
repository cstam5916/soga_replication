import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, to_dense_adj
from karateclub import Role2Vec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GraphEmbedding'))
from ge.models import Struc2Vec

class SOGALoss(nn.Module):
    def __init__(
        self,
        graph=None,
        mode='SOGA',
        lambda_local=1.0,
        lambda_role=0.5,
        role_k=None,
        embed_mode='Role2Vec',
    ):
        super().__init__()
        self.mode = mode
        self.lambda_local = lambda_local
        self.lambda_role = lambda_role

        self.register_buffer("P_local", torch.empty(0))
        self.register_buffer("P_role", torch.empty(0))

        if graph is not None:
            num_nodes = graph.num_nodes
            # Total number of edges in the graph for kappa
            num_edges = graph.edge_index.size(1)

            # 1. Local Adjacency (Local Neighbor Similarity)
            A = to_dense_adj(graph.edge_index, max_num_nodes=num_nodes).squeeze(0).float()
            A.fill_diagonal_(0)
            # Ensure it is symmetric and binary
            A = ((A + A.T) > 0).float()
            self.register_buffer("P_local", A)

            if (self.mode != "IMOnly"):
                G = to_networkx(graph, to_undirected=True)
                if embed_mode == 'Struc2Vec':
                    print("Fitting Struc2Vec...")
                    s2v_model = Struc2Vec(G, walk_length=10, num_walks=80, workers=1, verbose=0)
                    s2v_model.train(embed_size=64)
                    emb_dict = s2v_model.get_embeddings()
                    emb = torch.tensor(
                        [emb_dict[i] for i in range(num_nodes)], dtype=torch.float
                    )
                else:
                    r2v_model = Role2Vec(dimensions=64, walk_number=10, walk_length=80)
                    print("Fitting Role2Vec...")
                    r2v_model.fit(G)
                    emb = torch.tensor(r2v_model.get_embedding(), dtype=torch.float)
                emb = F.normalize(emb, p=2, dim=1)
                sim = emb @ emb.T
                
                mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
                upper_sims = sim[mask]
                
                kappa = num_edges if role_k is None else role_k
                kappa = min(kappa, upper_sims.numel())

                top_k_values, _ = torch.topk(upper_sims, k=kappa)
                threshold = top_k_values[-1]

                P_role = (sim >= threshold).float()
                P_role.fill_diagonal_(0) # Remove self-loops
                
                self.register_buffer("P_role", P_role)

    def forward(self, logits, edge_index):
        im_loss = 0
        sc_loss = 0

        if(self.mode != 'SCOnly'):
            log_q_cond = F.log_softmax(logits, dim=1)
            q_cond = log_q_cond.exp()
            q_marg = q_cond.mean(dim=0)
            q_marg = q_marg / q_marg.sum()

            im_loss = (
                self._conditional_entropy_loss(q_cond, log_q_cond)
                - self._marginal_entropy_loss(q_marg)
            )

        if (self.mode != 'IMOnly'):
            probs = F.softmax(logits, dim=1)
            sc_loss = self._structural_consistency_loss(probs)

        return im_loss + sc_loss

    def _conditional_entropy_loss(self, q_cond, log_q_cond):
        return -(q_cond * log_q_cond).sum(dim=1).mean()

    def _marginal_entropy_loss(self, q_marg):
        return -(q_marg * q_marg.clamp_min(1e-12).log()).sum()

    def _pairwise_bce_from_indicator(self, probs, P):
        """
        probs: [N, C]
        P: [N, N] binary indicator matrix
        """
        inner = probs @ probs.T         # [N, C], all pairwise inner products
        S = torch.sigmoid(inner).clamp(1e-12, 1 - 1e-12)

        # optional: exclude diagonal self-pairs
        off_diag = ~torch.eye(S.size(0), dtype=torch.bool, device=S.device)

        P = P[off_diag]
        S = S[off_diag]

        loss = -(P * torch.log(S) + (1 - P) * torch.log(1 - S)).mean()
        return loss

    def _structural_consistency_loss(self, probs):
        local_loss = self._pairwise_bce_from_indicator(probs, self.P_local)

        if self.P_role.numel() == 0:
            return self.lambda_local * local_loss

        role_loss = self._pairwise_bce_from_indicator(probs, self.P_role)

        return self.lambda_local * local_loss + self.lambda_role * role_loss