import torch
import torch.nn.functional as F
import torch.nn as nn
from .base_gnn import BaseGNNLayer
import torch


class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning with Multi-Head Attention
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(ReasonGNNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']

        # 多頭 attention ===
        self.num_heads = getattr(args, "num_heads", 1)
        self.attn_linears = nn.ModuleList(
            [nn.Linear(2 * entity_dim, 1) for _ in range(self.num_heads)]
        )

        self.use_posemb = args['pos_emb']
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i),
                                nn.Linear(in_features=2 * (self.num_ins) * entity_dim + entity_dim,
                                          out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        self.lin_m = nn.Linear(in_features=(self.num_ins) * entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        """
        初始化 reasoning 過程（建立圖結構 & entity embedding）
        """
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features

        # relation + query 向量
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            fact_val = F.relu((rel_linear(fact_rel) + pe) * fact_query)
        else:
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)

        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        # === Attention: 多頭 ===
        head_emb = torch.index_select(
            self.local_entity_emb.view(-1, self.entity_dim),
            dim=0,
            index=self.batch_heads
        )

        attn_outputs = []
        for attn_linear in self.attn_linears:
            attn_input = torch.cat([head_emb, fact_val], dim=-1)  # (edge_num, 2*dim)
            attn_score = attn_linear(attn_input)  # (edge_num, 1)
            attn_score = F.leaky_relu(attn_score)

            attn_weight = torch.exp(attn_score).view(-1, 1)
            attn_norm = torch.sparse.mm(self.fact2head_mat, attn_weight)
            attn_weight = attn_weight / (attn_norm[self.batch_heads] + 1e-6)

            attn_outputs.append(fact_val * attn_weight)

        fact_val = torch.mean(torch.stack(attn_outputs, dim=0), dim=0)

        # ========== 聚合 tail entity ==========
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)

    
        f2e_emb = F.relu(f2e_emb)

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)

        return neighbor_rep


    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        """
        Aggregates neighbor representations with Multi-Head Attention
        (tail → head, inverse edges)
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features_inv

        # relation + query 向量
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb_inv is not None:
            pe = pos_emb_inv(self.batch_rels)
            fact_val = F.relu((rel_linear(fact_rel) + pe) * fact_query)
        else:
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)

        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))

        # === Attention: 多頭 ===
        tail_emb = torch.index_select(
            self.local_entity_emb.view(-1, self.entity_dim),
            dim=0,
            index=self.batch_tails
        )

        attn_outputs = []
        for attn_linear in self.attn_linears:
            attn_input = torch.cat([tail_emb, fact_val], dim=-1)  # (edge_num, 2*dim)
            attn_score = attn_linear(attn_input)  # (edge_num, 1)
            attn_score = F.leaky_relu(attn_score)
            attn_weight = torch.exp(attn_score).view(-1, 1)
            attn_norm = torch.sparse.mm(self.fact2tail_mat, attn_weight)
            attn_weight = attn_weight / (attn_norm[self.batch_tails] + 1e-6)

            attn_outputs.append(fact_val * attn_weight)

        # 多頭: 平均聚合
        fact_val = torch.mean(torch.stack(attn_outputs, dim=0), dim=0)

        # ========== 聚合 head entity ==========
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)

        f2e_emb = F.relu(f2e_emb)
        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep


    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))

        neighbor_reps = []
        if self.use_posemb:
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else:
            pos_emb, pos_emb_inv = None, None

        # === 每個 instruction 聚合 ===
        for j in range(relational_ins.size(1)):
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:, j, :], rel_linear, pos_emb)
            neighbor_reps.append(neighbor_rep)
            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:, j, :], rel_linear, pos_emb_inv)
            neighbor_reps.append(neighbor_rep)

        # === 聚合鄰居 ===
        neighbor_reps = torch.cat(neighbor_reps, dim=2)

        # === DEBUG: 比較 TypeLayer 輸出 vs neighbor rep ===
        tl_norm = self.local_entity_emb.norm().item()
        nb_norm = neighbor_reps.norm().item()

        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)

        # === DEBUG: concat 之後整體 norm ===
        concat_norm = next_local_entity_emb.norm().item()

        self.local_entity_emb = torch.tanh(e2e_linear(self.linear_drop(next_local_entity_emb)))


        # === logits ===
        score_tp = self.score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)

        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp.masked_fill(answer_mask == 0, float('-inf'))

        self.last_answer_mask = answer_mask.detach().clone()

        current_dist = torch.zeros_like(score_tp)
        valid_mask = torch.any(answer_mask.bool(), dim=1)

        if valid_mask.any():
            current_dist[valid_mask] = self.softmax_d1(score_tp[valid_mask])
        if (~valid_mask).any():
            current_dist[~valid_mask] = torch.full_like(
                score_tp[~valid_mask], 1.0 / score_tp.size(1)
            )

        if return_score:
            return score_tp, current_dist, self.local_entity_emb
        return current_dist, self.local_entity_emb

