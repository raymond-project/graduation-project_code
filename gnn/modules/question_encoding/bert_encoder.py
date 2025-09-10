
import torch.nn.functional as F
import torch.nn as nn
import os
import torch 

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


from transformers import AutoModel, AutoTokenizer #DistilBertModel, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from torch.nn import LayerNorm
import warnings
warnings.filterwarnings("ignore")


from .base_encoder import BaseInstruction


class BERTInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word, model, constraint=False):
        super(BERTInstruction, self).__init__(args, constraint)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.constraint = constraint
        
        entity_dim = self.entity_dim
        self.model = model
        
        
        if model == 'biobert-v1.1':
            model_path = os.path.join('/home/st426/system/GNN-RAG/gnn/bert_model', model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.pretrained_weights = model_path
            word_dim = 768    # 來源於 config中的 "hidden_size" 
            
        elif model == 'scibert-nli':
            model_path = os.path.join('/home/st426/system/GNN-RAG/gnn/bert_model', model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.pretrained_weights = model_path
            word_dim = 768    # 來源於 config中的 "hidden_size
        elif model == 'ClinicalBERT':
            model_path = os.path.join('/home/st426/system/GNN-RAG/gnn/bert_model', model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.pretrained_weights = model_path
            word_dim = 768    # 來源於 config中的 "hidden_size"  

        #elif model == '':  看之後會不會 有其他bert模型我固定都放在GNN-RAG/gnn/bert_model下 依據模型名稱為檔名
        else: 
            raise ValueError(f"沒有這模型: {model}")
        
        
        
        #pad_val 是取得 tokenizer 的 padding token ID，用來做 attention mask
        #word_dim 是語言模型輸出向量的維度，通常為 768（或 384, 1024 等）from config
        self.pad_val = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.word_dim = word_dim
        print('word_dim', self.word_dim)
        
        
    
        #這兩層是用來後續進行注意力計算與融合（如 query-entity matching）用途的 linear 層。 
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        
        
        #為每個 instruction 步驟加上獨立的線性層（可想成用來更新 query 向量的模組）
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            
            
        #把從 BERT 輸出的 word-level embedding 轉成模型內部用的向量空間（從 word_dim 映射到 entity_dim
        self.question_emb = nn.Linear(in_features=word_dim, out_features=entity_dim)
        
        
        
        #載入模型本體 --lm 有bert的話這就會True
        if not self.constraint:
            self.encoder_def()







#以下為我還沒看細節
#encoder_def = 把模型載好
#encode_question = 拿模型來編碼問題文字
    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        entity_dim = self.entity_dim
        self.node_encoder = AutoModel.from_pretrained(self.pretrained_weights)
        print('總共的 Params', sum(p.numel() for p in self.node_encoder.parameters()))
        
        
        
        #應該是不用用到LM params 因為預先訓練模型就很好用
        
        #只訓練你 GNN 模型 
        if self.lm_frozen == 1:
            print('凍結 LM params')
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        
        
        
        
        #整個 BERT/BioBERT 模型會跟著 GNN 一起微調
        #如果用這fine-tuing後應該可以適應下游 但是之後要記得要重新git一次模型
        else:
            for param in self.node_encoder.parameters():
                param.requires_grad = True
            print('不凍結 LM params')
            
            

    def encode_question(self, query_input, store=True):
        outputs = self.node_encoder(**query_input)
        hidden_emb = outputs.last_hidden_state  # (batch, seq_len, 768)
        hidden_emb = torch.nan_to_num(hidden_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # log question_emb 的權重 norm
        if hasattr(self, "question_emb"):
            print("[DEBUG][encode_question] question_emb weight norm:",
                self.question_emb.weight.norm().item())

        # CLS pooling
        cls_emb = hidden_emb[:, 0:1, :]  # (batch, 1, 768)
        query_node_emb = self.question_emb(cls_emb)  # (batch, 1, entity_dim)
        query_node_emb = torch.nan_to_num(query_node_emb, nan=0.0, posinf=0.0, neginf=0.0)

        if store:
            # ✅ 存下來，給 get_instruction 用
            self.query_hidden_emb = self.question_emb(hidden_emb)
            self.query_node_emb = query_node_emb
            self.query_mask = query_input["attention_mask"].float()

            print("[DEBUG][encode_question] store=True | hidden_emb:", hidden_emb.shape,
                "query_hidden_emb:", self.query_hidden_emb.shape,
                "query_node_emb:", self.query_node_emb.shape)
            return self.query_hidden_emb, self.query_node_emb
        else:
            # 🚨 relation encoding: 只回 CLS vector
            print("[DEBUG][encode_question] store=False | return query_node_emb:", query_node_emb.shape)
            return query_node_emb.squeeze(1)

