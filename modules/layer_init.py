import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TypeLayer(nn.Module):
    """
    Sparse version GAT layer with mean aggregation (degree normalization) + debug
    """

    def __init__(self, in_features, out_features, linear_drop, device, norm_rel):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device
        self.norm_rel = norm_rel

        self._register_hooks()

    def _register_hooks(self):
        def bw_hook(module, grad_input, grad_output):
            try:
                if grad_output[0] is not None:
                    g = grad_output[0]
                    print(f"[DEBUG][TypeLayer][BWD] grad_output "
                          f"range=({g.min().item():.6f}, {g.max().item():.6f}), "
                          f"mean={g.mean().item():.6f}, "
                          f"std={g.std().item():.6f}, "
                          f"has_nan={torch.isnan(g).any().item()}")
            except Exception as e:
                print(f"[DEBUG][TypeLayer][BWD] hook error: {e}")
        self.kb_self_linear.register_full_backward_hook(bw_hook)

    def forward(self, local_entity, edge_list, rel_features):
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()

        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)

        if self.norm_rel:
            val_one = torch.FloatTensor(weight_rel_list).to(self.device)
        else:
            val_one = torch.ones_like(batch_ids).float().to(self.device)

        # === Step 1: relation → value ===
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        fact_rel = torch.clamp(fact_rel, -1e3, 1e3)
        fact_val = self.kb_self_linear(fact_rel)

        print(f"[DEBUG][TypeLayer] fact_val shape={fact_val.shape}, "
              f"range=({fact_val.min().item():.6f}, {fact_val.max().item():.6f}), "
              f"norm={fact_val.norm().item():.4f}")

        # === Step 2: sparse aggregation ===
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        # Degree
        deg_head = torch.sparse.sum(fact2head_mat, dim=1).to_dense()
        deg_tail = torch.sparse.sum(fact2tail_mat, dim=1).to_dense()
        deg_total = deg_head + deg_tail
        deg_total = deg_total.clamp(min=1.0)  # 避免除 0

        print(f"[DEBUG][TypeLayer] degree stats | min={deg_total.min().item():.2f}, "
              f"max={deg_total.max().item():.2f}, mean={deg_total.mean().item():.2f}")

        # Mean aggregation
        f2e_emb = torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val)
        f2e_emb = f2e_emb / deg_total.unsqueeze(-1)   # <-- 均值聚合
        print(f"[DEBUG][TypeLayer] f2e_emb (pre-relu) range=({f2e_emb.min().item():.6f}, {f2e_emb.max().item():.6f}), "
              f"mean={f2e_emb.mean().item():.6f}, std={f2e_emb.std().item():.6f}")

        # === Step 3: activation + safety ===
        f2e_emb = F.relu(f2e_emb)
        f2e_emb = torch.nan_to_num(f2e_emb, nan=0.0, posinf=1e4, neginf=-1e4)

        # reshape
        f2e_emb = f2e_emb.view(batch_size, max_local_entity, self.out_features)
        f2e_emb = torch.nan_to_num(f2e_emb, nan=0.0, posinf=1e4, neginf=-1e4)

        assert not torch.isnan(f2e_emb).any(), "[ERROR][TypeLayer] f2e_emb 出現 NaN"

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
