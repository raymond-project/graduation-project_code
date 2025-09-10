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
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)
        

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        #self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        #self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        #self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        #self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
   
    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
        if self.rel_texts is None:
            # === case 1: 沒有關係文字，用隨機初始化的 embedding ===
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight

            print("[DEBUG][get_rel_feature] raw rel_embedding:",
                "shape:", rel_features.shape,
                "nan:", torch.isnan(rel_features).any().item(),
                "inv_nan:", torch.isnan(rel_features_inv).any().item())

            # 線性投影到 entity_dim
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)

        else:
            # === case 2: 有關係文字 ===
            rel_features = self.rel_features
            rel_features_inv = self.rel_features_inv

            print("[DEBUG][get_rel_feature] rel_features:",
                "shape:", rel_features.shape,
                "nan:", torch.isnan(rel_features).any().item(),
                "min:", rel_features.min().item(),
                "max:", rel_features.max().item())
            print("[DEBUG][get_rel_feature] rel_features_inv:",
                "shape:", rel_features_inv.shape,
                "nan:", torch.isnan(rel_features_inv).any().item())

            # 🔒 NaN 保護
            rel_features = torch.nan_to_num(rel_features, nan=0.0, posinf=1e4, neginf=-1e4)
            rel_features_inv = torch.nan_to_num(rel_features_inv, nan=0.0, posinf=1e4, neginf=-1e4)

            # === 決定是否進 self_att_r ===
            if rel_features.dim() == 3:
                mask = (self.rel_texts != self.instruction.pad_val).float()
                rel_features = self.self_att_r(rel_features, mask)
                rel_features_inv = self.self_att_r(rel_features_inv, mask)
                print("[DEBUG][get_rel_feature] after attn:",
                    "rel_nan:", torch.isnan(rel_features).any().item(),
                    "inv_nan:", torch.isnan(rel_features_inv).any().item())
            else:
                print("[DEBUG][get_rel_feature] skip self_att_r (CLS only)")

        return rel_features, rel_features_inv






    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv  = self.get_rel_feature()
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
                                   query_entities=query_entities)

    def calc_loss_label(self, raw_score, teacher_dist, label_valid):
        """
        使用 BCEWithLogitsLoss，直接吃 raw_score (未經 sigmoid)，避免梯度消失。
        """
        bce = nn.BCEWithLogitsLoss(reduction='none')
        tp_loss = bce(raw_score, teacher_dist)   # [batch_size, num_entity]
        tp_loss = tp_loss * label_valid          # mask 無效樣本
        cur_loss = torch.sum(tp_loss) / raw_score.size(0)
        return cur_loss


    
    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """
        # unpack batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val
            query_mask = (q_input != pad_val).float()
        else:
            query_mask = (q_input != self.num_word).float()

        # === 初始化 reasoning ===
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                        kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)

        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins

        self.dist_history.append(self.curr_dist)

        # === BFS + GNN reasoning ===
        actions_per_step = [] 
        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            for j in range(self.num_gnn):
                raw_score, self.curr_dist, global_rep = self.reasoning(
                    self.curr_dist, relation_ins, step=j, return_score=True
                )

                # Debug
                print(f"[DEBUG][ReaRev] step={j} raw_score stats | "
                    f"min={raw_score.min().item():.4f}, "
                    f"max={raw_score.max().item():.4f}, "
                    f"mean={raw_score.mean().item():.4f}, "
                    f"std={raw_score.std().item():.4f}")

            # 存 (logits, 分布)
            self.dist_history.append((raw_score, self.curr_dist))

            # Instruction Updates
            qs = []
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)

        # === 預測 ===
        raw_score, pred_dist_now = self.dist_history[-1]   # logits, 分布
        raw_score = torch.clamp(raw_score, -20, 20)
        
        pred_dist = torch.sigmoid(raw_score)               # 機率分布

        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()

        #  用 raw_score + BCEWithLogitsLoss
        loss = self.calc_loss_label(
            raw_score=raw_score,
            teacher_dist=answer_dist,
            label_valid=case_valid
        )

        pred = torch.max(pred_dist, dim=1)[1]
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"[FATAL] Non-finite loss detected! "
                f"logits range=({raw_score.min().item():.4f}, {raw_score.max().item():.4f}), "
                f"mean={raw_score.mean().item():.4f}, std={raw_score.std().item():.4f}"
            )

        if training:
            # ✅ train 階段只回傳 h1/f1
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            # ✅ eval 階段回傳完整 actions trace
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [actions_per_step, h1.tolist(), f1.tolist()]

        # === 回傳五個值 ===
        print(f"[DEBUG][forward] logits range=({raw_score.min().item():.4f}, {raw_score.max().item():.4f}) "
            f"mean={raw_score.mean().item():.4f} std={raw_score.std().item():.4f}")

        return loss, raw_score, pred_dist, pred, tp_list


