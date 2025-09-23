import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word) # 呼叫子副類別 BaseModel 的 __init__ 方法
        self.norm_rel = args['norm_rel']
        self.layers(args)

        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

        self.private_module_def(args, num_entity, num_relation) #轉向量 將 BERTInstruction 來處理輸入的文字 

        self.to(self.device)
        self.lin = nn.Linear(3 * self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []

       
        self.margin = args.get('margin', 1.0)

        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))

    def layers(self, args):
        entity_dim = self.entity_dim
        self.linear_dropout = args['linear_dropout']

        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim) 

        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(
                in_features=entity_dim,
                out_features=entity_dim,
                linear_drop=self.linear_drop,
                device=self.device,
                norm_rel=self.norm_rel
            )

        self.self_att_r = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(
                local_entity=local_entity,
                edge_list=kb_adj_mat,
                rel_features=rel_features
            )
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)

        return local_entity_emb

    def get_rel_feature(self):
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = torch.nan_to_num(self.rel_features, nan=0.0, posinf=1e4, neginf=-1e4)
            rel_features_inv = torch.nan_to_num(self.rel_features_inv, nan=0.0, posinf=1e4, neginf=-1e4)
        return rel_features, rel_features_inv

    def private_module_def(self, args, num_entity, num_relation):
        # 初始化對於 Rearev 作用 init__
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg) # 處理 message passing  
        self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])   #  負責把問題文字轉成向量
            # 開頭用了class ReaRev(BaseModel): 所以會去呼叫 BaseModel 的 init_ 的 self.word_embedding, self.num_word
            
            
    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)     #利用 self.instruction = BERTInstruction( 編碼
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist

        self.reasoning.init_reason(
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            local_entity_emb=self.local_entity_emb,
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities
        )

    def contrastive_loss_topk(self, raw_score, teacher_dist, label_valid, margin=1.0, k=5):
        """
        Top-k Contrastive Loss:
        不只取最難的一個負樣本，而是取前 k 個最難的負樣本，取平均來計算 loss。
        這樣可以讓模型同時壓制多個難的負樣本，避免 loss 忽高忽低。
        """
        # 標記正樣本
        pos_mask = (teacher_dist > 0).float()           # [B, E]
        has_pos = (pos_mask.sum(dim=1) > 0).float()     # [B]

        # 正樣本平均分數
        pos_score = (raw_score * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)

        # 負樣本 (遮掉正樣本，剩下全是負樣本)
        neg_logit = raw_score.masked_fill(pos_mask.bool(), -1e9)

        # 取前 k 個最難負樣本
        topk_neg, _ = torch.topk(neg_logit, k=min(k, neg_logit.size(1)), dim=1)

        # top-k 取平均 (讓模型壓制一群難的負樣本，而不是只壓制最難的那一個)
        hard_neg = topk_neg.mean(dim=1)

        # margin-based loss
        con = torch.relu(margin - pos_score + hard_neg)  # [B]

        # 只對有效樣本計算
        valid = (label_valid.squeeze(1) * has_pos)
        con_loss = (con * valid).sum() / valid.sum().clamp(min=1.0)
        return con_loss


    def forward(self, batch, training=False):
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)  #
        """ 
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val
            query_mask = (q_input != pad_val).float()
        else:
            query_mask = (q_input != self.num_word).float()
        """

        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input) #來自於 bert 的 tokenized query: shape [batch, seq_len] 

        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins

        self.dist_history.append(self.curr_dist)

        actions_per_step = []
        
        
        #每次迭代時 拿 self.instruction_list, self.attn_list = self.instruction(q_input)  
        # 把 query instructions 存下來，後面會繼續更新 
        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            for j in range(self.num_gnn):
                raw_score, self.curr_dist, global_rep = self.reasoning(
                    self.curr_dist, relation_ins, step=j, return_score=True
                )

            self.dist_history.append((raw_score, self.curr_dist))

            qs = []
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(
                    self.instruction.instructions[j].squeeze(1),  
                    self.local_entity_emb,                        
                    query_entities,                               
                    self.reasoning.local_entity_mask              
                )

                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)

        raw_score, pred_dist_now = self.dist_history[-1]
        raw_score = torch.clamp(raw_score, -20, 20)
        pred_dist = torch.sigmoid(raw_score)

        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()

        
        loss = self.contrastive_loss_topk(
            raw_score=raw_score,
            teacher_dist=answer_dist,
            label_valid=case_valid,
            margin=self.margin,
            k=5   
        )


        if hasattr(self.reasoning, "local_entity_mask"):
            self.last_answer_mask = self.reasoning.local_entity_mask.detach().clone()

        pred = torch.max(pred_dist, dim=1)[1]

        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [actions_per_step, h1.tolist(), f1.tolist()]

        return loss, raw_score, pred_dist, pred, tp_list, {
            "contrastive": loss.item()
        }
