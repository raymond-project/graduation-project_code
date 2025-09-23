import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
import random
import warnings
import pickle
warnings.filterwarnings("ignore")
from modules.question_encoding.tokenizers import LSTMTokenizer#, BERTTokenizer
from transformers import AutoTokenizer
import time

import os


class BasicDataLoader(object):
    """ 
    Basic Dataloader contains all the functions to read questions and KGs from json files and
    create mappings between global entity ids and local ids that are used during GNN updates.
    """

    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        self.tokenize = tokenize.lower() 
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()
        
    def _extract_edges_from_tuples(self, sample, g2l, sample_id):
        """
        從一個 sample 的 subgraph 中提取 edges (head, relation, tail)，
        並寫進 self.kb_fact_rels。
        """
        head_list, rel_list, tail_list = [], [], []

        for tpl in sample['subgraph']['tuples']:
            sbj, rel, obj = tpl
            try:
                if isinstance(sbj, dict) and 'text' in sbj:
                    head = g2l[self.entity2id[sbj['text']]]
                    rel = self.relation2id[rel['text']]
                    tail = g2l[self.entity2id[obj['text']]]
                else:
                    head = g2l[self.entity2id[sbj]]
                    rel = self.relation2id[rel]
                    tail = g2l[self.entity2id[obj]]
            except:
                head = g2l[sbj]
                try:
                    rel = int(rel)
                except:
                    rel = self.relation2id[rel]
                tail = g2l[obj]

            # === 正向邊 ===
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            self.kb_fact_rels[sample_id, len(head_list) - 1] = rel

            # === 逆向邊 ===
            if self.use_inverse_relation:
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)
                self.kb_fact_rels[sample_id, len(head_list) - 1] = rel + len(self.relation2id)


        return head_list, rel_list, tail_list




    def _load_file(self, config, data_type="train"):

        """
        Loads lines (questions + KG subgraphs) from json files.
        """
        
        data_file = config['data_folder'] + data_type + ".json"
        self.data_file = data_file
        print('loading data from', data_file)
        self.data_type = data_type
        self.data = []
        skip_index = set()
        index = 0

        with open(data_file) as f_in:
            for line in tqdm(f_in):
                if index == config['max_train'] and data_type == "train": break  #break if we reach max_question_size
                line = json.loads(line)
                
                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                index += 1

        print("skip", skip_index)
        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _load_data(self):

        """
        Creates mappings between global entity ids and local entity ids that are used during GNN updates.
        """

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        self._prepare_data()

    def _parse_args(self, config, word2id, relation2id, entity2id):

        """
        Builds necessary dictionaries and stores arguments.
        """
        self.data_eff = config['data_eff']
        self.data_name = config['name']

        if 'use_inverse_relation' in config:
            self.use_inverse_relation = config['use_inverse_relation']
        else:
            self.use_inverse_relation = False
        if 'use_self_loop' in config:
            self.use_self_loop = config['use_self_loop']
        else:
            self.use_self_loop = False

        self.rel_word_emb = config['relation_word_emb']
        #self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        print('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        #print(">>> [DEBUG] entity2id keys sample:", list(self.entity2id.keys())[:10])
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        self.q_type = config['q_type']

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    
    def get_quest(self, training=False):
        q_list = []
        
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            # id2word = self.id2word
            # for i in range(self.max_query_word):
            #     if self.query_texts[sample_id, i] in id2word:
            #         tp_str += id2word[self.query_texts[sample_id, i]] + " "
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        if self.tokenize == 'lstm':
            id2word = self.id2word
            tp_str = ""
            for i in range(self.max_query_word):
                if np_array_x[i] in id2word:
                    tp_str += id2word[np_array_x[i]] + " "
        else:
            tp_str = ""
            words = self.tokenizer.convert_ids_to_tokens(np_array_x)
            for w in words:
                if w not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                    tp_str += w + " "
        return tp_str
    

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        max_count = 0
        for line in self.data:
            word_list = line["question"].split(' ')
            max_count = max(max_count, len(word_list))

        if self.rel_word_emb:
            self.build_rel_words(self.tokenize)
        else:
            self.rel_texts = None
            self.rel_texts_inv = None
            self.ent_texts = None

        self.max_query_word = max_count

        # build tokenizers
        if self.tokenize == 'lstm':
            self.num_word = len(self.word2id)
            self.tokenizer = LSTMTokenizer(self.word2id, self.max_query_word)
            self.query_texts = np.full((self.num_data, self.max_query_word), self.num_word, dtype=int)
        elif self.tokenize == 'biobert-v1.1':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/biobert-v1.1"
            self.max_query_word = max_count + 2
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            self.num_word = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id
            self.query_texts = np.full((self.num_data, self.max_query_word), pad_id, dtype=int)
        elif self.tokenize == 'scibert-nli':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/scibert-nli"
            self.max_query_word = max_count + 2
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            self.num_word = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id
            self.query_texts = np.full((self.num_data, self.max_query_word), pad_id, dtype=int)
        elif self.tokenize == 'clinicalbert':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/ClinicalBERT"
            self.max_query_word = max_count + 2
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            self.num_word = len(self.tokenizer)
            pad_id = self.tokenizer.pad_token_id
            self.query_texts = np.full((self.num_data, self.max_query_word), pad_id, dtype=int)

        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                continue

            # === entities ===
            tp_set = set()
            seed_list = []
            key_ent = 'entities_cid' if 'entities_cid' in sample else 'entities'
            for entity in sample[key_ent]:
                try:
                    if isinstance(entity, dict) and 'text' in entity:
                        global_entity = self.entity2id[entity['text']]
                    else:
                        global_entity = self.entity2id[entity]
                    global_entity = self.entity2id[entity['text']]
                except:
                    global_entity = entity

                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)

            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)

            for global_entity, local_entity in g2l.items():
                if self.data_name != 'cwq':
                    if local_entity not in tp_set:
                        self.candidate_entities[next_id, local_entity] = global_entity
                else:
                    self.candidate_entities[next_id, local_entity] = global_entity

            # === relations in local KB ===
            head_list, rel_list, tail_list = self._extract_edges_from_tuples(sample, g2l, next_id)

            if not self.data_eff:
                self.kb_adj_mats[next_id] = (
                    np.array(head_list, dtype=int),
                    np.array(rel_list, dtype=int),
                    np.array(tail_list, dtype=int)
                )

            # === seed distribution ===
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            assert np.sum(self.seed_distribution[next_id]) > 0.0

            # === tokenize question ===
            if self.tokenize == 'lstm':
                self.query_texts[next_id] = self.tokenizer.tokenize(sample['question'])
            else:
                tokens = self.tokenizer.encode_plus(
                    text=sample['question'],
                    max_length=self.max_query_word,
                    padding='max_length',
                    return_attention_mask=False,
                    truncation=True
                )
                self.query_texts[next_id] = np.array(tokens['input_ids'])

            # === answers ===
            answer_list = []
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                answer_ent = int(answer[keyword])
                answer_list.append(answer_ent)
                if answer_ent in g2l:
                    self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list

            next_id += 1

        # summary
        num_no_query_ent = sum(1 for i in range(next_id) if num_query_entity[i] == 0)
        num_one_query_ent = sum(1 for i in range(next_id) if num_query_entity[i] == 1)
        num_multiple_ent = sum(1 for i in range(next_id) if num_query_entity[i] > 1)
        print(f"{next_id} cases in total, {num_no_query_ent} without query entity, "
            f"{num_one_query_ent} with single query entity, {num_multiple_ent} with multiple query entities")


        
    def build_rel_words(self, tokenize):
        """ 
        Tokenizes relation surface forms.
        """

        max_rel_words = 0
        rel_words = []
        if 'metaqa' in self.data_file:
            for rel in self.relation2id:   #  這裡有改
                words = rel.split(' ')    #  原來是 words = rel.split('_') 
                max_rel_words = max(len(words), max_rel_words)
                rel_words.append(words)
            #print(rel_words)
        else:
            for rel in self.relation2id:
                rel = rel.strip()
                words = rel.split(' ')  
                try:
                    #words = fields[-2].split('_') + fields[-1].split('_')
                    max_rel_words = max(len(words), max_rel_words)
                    rel_words.append(words)
                    #print(rel, words)
                except:
                    words = ['UNK']
                    rel_words.append(words)
                    pass
                #words = fields[-2].split('_') + fields[-1].split('_')
            
        self.max_rel_words = max_rel_words
        if tokenize == 'lstm':
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), len(self.word2id), dtype=int)
            for rel_id,tokens in enumerate(rel_words):
                for j, word in enumerate(tokens):
                    if j < self.max_rel_words:
                            if word in self.word2id:
                                self.rel_texts[rel_id, j] = self.word2id[word]
                                self.rel_texts_inv[rel_id, j] = self.word2id[word]
                            else:
                                self.rel_texts[rel_id, j] = len(self.word2id)
                                self.rel_texts_inv[rel_id, j] = len(self.word2id)
        elif tokenize == 'biobert-v1.1':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/biobert-v1.1"
        elif tokenize == 'scibert-nli':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/scibert-nli"
        elif tokenize == 'clinicalbert':
            tokenizer_name = "/home/st426/system/GNN-RAG/gnn/bert_model/ClinicalBERT"
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            pad_val = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            self.rel_texts = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            self.rel_texts_inv = np.full((self.num_kb_relation + 1, self.max_rel_words), pad_val, dtype=int)
            
            for rel_id,words in enumerate(rel_words):

                tokens =  tokenizer.encode_plus(text=' '.join(words), max_length=self.max_rel_words, \
                    padding="max_length", return_attention_mask = False, truncation=True)
                tokens_inv =  tokenizer.encode_plus(text=' '.join(words[::-1]), max_length=self.max_rel_words, \
                    padding="max_length", return_attention_mask = False, truncation=True)
                self.rel_texts[rel_id] = np.array(tokens['input_ids'])
                self.rel_texts_inv[rel_id] = np.array(tokens_inv['input_ids'])


        
        #print(rel_words)
        #print(len(rel_words), len(self.relation2id))
        assert len(rel_words) == len(self.relation2id)
        #print(self.rel_texts, self.max_rel_words)

    def create_kb_adj_mats(self, sample_id):

        """
        Re-build local adj mats if we have data_eff == True (they are not pre-stored).
        """
        sample = self.data[sample_id]
        g2l = self.global2local_entity_maps[sample_id]
        
        # build connection between question and entities in it
        # relations in local KB
        head_list = []
        rel_list = []
        tail_list = []
        for tpl in sample['subgraph']['tuples']:
            sbj, rel, obj = tpl
            try:
                if isinstance(sbj, dict) and 'text' in sbj:
                    head = g2l[self.entity2id[sbj['text']]]
                    rel = self.relation2id[rel['text']]
                    tail = g2l[self.entity2id[obj['text']]]
                else:
                    head = g2l[self.entity2id[sbj]]
                    rel = self.relation2id[rel]
                    tail = g2l[self.entity2id[obj]]
            except:
                head = g2l[sbj]
                try:
                    rel = int(rel)
                except:
                    rel = self.relation2id[rel]
                tail = g2l[obj]

            # === 正向邊 ===
            head_list.append(head)
            rel_list.append(rel)
            tail_list.append(tail)
            self.kb_fact_rels[sample_id, len(head_list)-1] = rel
  # 保證 index 對齊

            # === 逆向邊 ===
            if self.use_inverse_relation:
                head_list.append(tail)
                rel_list.append(rel + len(self.relation2id))
                tail_list.append(head)
                self.kb_fact_rels[sample_id, len(head_list)-1] = rel + len(self.relation2id)



        return self._extract_edges_from_tuples(sample, g2l, sample_id)

    
    def _build_fact_mat(self, sample_ids, fact_dropout):
        """
        Creates local adj mats that contain entities, relations, and structure.
        固定長度: batch_size × max_facts
        """
        batch_heads, batch_rels, batch_tails, batch_ids = [], [], [], []

        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            if self.data_eff:
                head_list, rel_list, tail_list = self.create_kb_adj_mats(sample_id)
            else:
                head_list, rel_list, tail_list = self.kb_adj_mats[sample_id]

            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[:num_keep_fact]

            for idx in mask_index:
                batch_heads.append(head_list[idx] + index_bias)
                batch_tails.append(tail_list[idx] + index_bias)
                batch_rels.append(rel_list[idx])
                batch_ids.append(i)

            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                for ent in range(num_ent_now):
                    batch_heads.append(ent + index_bias)
                    batch_tails.append(ent + index_bias)
                    batch_rels.append(self.num_kb_relation - 1)
                    batch_ids.append(i)

        # -------------------------------
        # 🔥 固定長度處理
        expected_facts = len(sample_ids) * self.max_facts
        cur_facts = len(batch_heads)

        if cur_facts > expected_facts:
            # truncate
            batch_heads = batch_heads[:expected_facts]
            batch_rels  = batch_rels[:expected_facts]
            batch_tails = batch_tails[:expected_facts]
            batch_ids   = batch_ids[:expected_facts]
        elif cur_facts < expected_facts:
            # padding
            pad_len = expected_facts - cur_facts
            batch_heads.extend([0] * pad_len)
            batch_rels.extend([self.num_kb_relation - 1] * pad_len)
            batch_tails.extend([0] * pad_len)
            batch_ids.extend([0] * pad_len)

        # numpy 化
        batch_heads = np.array(batch_heads, dtype=int)
        batch_rels = np.array(batch_rels, dtype=int)
        batch_tails = np.array(batch_tails, dtype=int)
        batch_ids = np.array(batch_ids, dtype=int)
        fact_ids = np.arange(len(batch_heads), dtype=int)

        head_count = Counter(batch_heads)
        weight_list = [1.0 / head_count[h] for h in batch_heads]

        head_rels_batch = list(zip(batch_heads, batch_rels))
        head_rels_count = Counter(head_rels_batch)
        weight_rel_list = [1.0 / head_rels_count[(h, r)] for (h, r) in head_rels_batch]


        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list





    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            if 'entities_cid' in sample:
                self._add_entity_to_map(self.entity2id, sample['entities_cid'], g2l)
            else:
                self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            #self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps



    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        #print(entities)
        #print(entity2id)
        for entity_global_id in entities:
            try:
                if isinstance(entity_global_id, dict) and 'text' in entity_global_id:
                    ent = entity2id[entity_global_id['text']]
                else:
                    ent = entity2id[entity_global_id]
                if ent not in g2l:
                    g2l[ent] = len(g2l)
            except:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        
        return q_input

    



class SingleDataLoader(BasicDataLoader):
    """
    Single Dataloader creates training/eval batches during KGQA.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)     #總共有幾個實體被納入這次訓練使用 來源entities.txt 計算行數拿來index 
    return word2id

def load_dict_int(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[int(word)] = int(word)
    return word2id

def load_data(config, tokenize):

    """
    Creates train/val/test dataloaders (seperately).
    """
    if 'sr-cwq' in config['data_folder']:
        entity2id = load_dict_int(config['data_folder'] + config['entity2id'])
    else:
        entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    #print("[DEBUG] relation2id:", relation2id)
    
    
    if config["is_eval"]:
        train_data = None
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = test_data.num_word
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = train_data.num_word
    relation_texts = test_data.rel_texts
    relation_texts_inv = test_data.rel_texts_inv
    entities_texts = None
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data, #test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": num_word,
        "rel_texts": relation_texts,
        "rel_texts_inv": relation_texts_inv,
        "ent_texts": entities_texts
    }
    
    return dataset


if __name__ == "__main__":
    st = time.time()
    #args = get_config()
    load_data(args)
    
