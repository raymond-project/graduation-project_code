#  Graph Attention Layer (GAT) 簡化版 Graph Attention Layer (GAT)，
# 主要透過 關係特徵 (relation features) 來生成實體 (entity) 向量。
# 但因為 針對GAT 原先是針對無向圖所以前面我用了矩陣與反矩陣來達到無向邊
""" 
ReaRev
  └─ ReasonGNNLayer
        ├─ BaseGNNLayer  (建立圖的稀疏矩陣)
        └─ TypeLayer     (用稀疏矩陣 + relation feature 做 entity 更新)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TypeLayer(nn.Module):
    def __init__(self, in_features, out_features, linear_drop, device, norm_rel):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device
        self.norm_rel = norm_rel
        self.last_fact_val = None
        # === 新增 Attention 參數 ===
        self.attn_w = nn.Linear(out_features * 2, 1, bias=False)  # concat(node, relation)
        self.leakyrelu = nn.LeakyReLU(0.2)



    def forward(self, local_entity, edge_list, rel_features):
            batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list = edge_list
            batch_size, max_local_entity = local_entity.size()
            num_fact = len(fact_ids)

            # relation → value 
            fact_rel = torch.index_select(rel_features, dim=0, index=torch.LongTensor(batch_rels).to(self.device)) 
            fact_val = self.kb_self_linear(fact_rel)# [num_fact, out_features] 
                                                    #  投影，對齊維度 例如 relation 是 200 維  entity 是 150 維 無法相加聚合

            # === Step 2: Attention score 計算 ===
            # head/tail entity 的 ID → 映射成 index
            heads = torch.LongTensor(batch_heads).to(self.device)
            tails = torch.LongTensor(batch_tails).to(self.device)

            # flatten entity index (因為 batch*max_local_entity 展開過)
            head_idx = heads + torch.LongTensor(batch_ids).to(self.device) * max_local_entity
            tail_idx = tails + torch.LongTensor(batch_ids).to(self.device) * max_local_entity

            # 取出 head/tail 節點的 one-hot index，這裡假設初始沒有 embedding，就只用 relation 特徵來做注意力
            # 如果你已經有 local_entity embedding，可以把 head_emb/tail_emb 傳進來替代
            head_emb = fact_val   # 暫時用 fact_val 當 query
            tail_emb = fact_val

            # attention input = concat(relation feature, relation value)
            attn_input = torch.cat([fact_val, fact_rel], dim=-1)  # [num_fact, 2*out_features]
            attn_score = self.attn_w(attn_input)  # [num_fact, 1]
            attn_score = self.leakyrelu(attn_score)

            # normalize
            attn_score = torch.exp(attn_score)
            attn_score = attn_score / (torch.sum(attn_score, dim=0, keepdim=True) + 1e-9)

            #  Weighted aggregation 
            fact2tail = torch.stack([tail_idx, torch.arange(num_fact).to(self.device)])
            fact2head = torch.stack([head_idx, torch.arange(num_fact).to(self.device)])

            fact2tail_mat = torch.sparse.FloatTensor(fact2tail, attn_score.squeeze(-1), (batch_size * max_local_entity, num_fact)).to(self.device)
            fact2head_mat = torch.sparse.FloatTensor(fact2head, attn_score.squeeze(-1), (batch_size * max_local_entity, num_fact)).to(self.device)

            f2e_emb = torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val)
            f2e_emb = F.elu(f2e_emb)

            f2e_emb = f2e_emb.view(batch_size, max_local_entity, self.out_features)
            return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)
