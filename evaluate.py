from tqdm import tqdm
tqdm.monitor_iterval = 0
import torch
import numpy as np
import math, os
import json
import pickle

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)


def f1_and_hits(answers, candidate2prob, id2entity, entity2name, eps=0.5):
    ans = []
    retrieved = []
    for a in answers:
        if entity2name is None:
            ans.append(id2entity[a])
        else:
            ans.append(entity2name[id2entity[a]])
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    for c, prob in cand_list:
        if entity2name is None:
            retrieved.append((id2entity[c], prob))
        else:
           retrieved.append((entity2name[id2entity[c]], prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if correct > 0:
        em = 1
    else:
        em = 0
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0, 1.0, 0, retrieved, ans  # precision, recall, f1, hits, em
        else:
            return 0.0, 1.0, 0.0, 1.0, 1.0, 1, retrieved , ans # precision, recall, f1, hits, em
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits, hits, 2, retrieved , ans # precision, recall, f1, hits, em
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits, em, 3, retrieved, ans


class Evaluator:

    def __init__(self, args, model, entity2id, relation2id, device):
        self.model = model
        self.args = args
        self.eps = args['eps']
        self.model_name = args["model_name"]
        
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        self.id2entity = id2entity

        self.entity2name = None
        if 'sr-' in args["data_folder"]:
            file = open('ent2id.pickle', 'rb')
            self.entity2name = list((pickle.load(file)).keys())
            file.close()

            
        id2relation = {idx: relation for relation, idx in relation2id.items()}
        num_rel_ori = len(relation2id)

        if 'use_inverse_relation' in args:
            self.use_inverse_relation = args['use_inverse_relation']
            if self.use_inverse_relation:
                for i in range(len(id2relation)):
                    id2relation[i + num_rel_ori] = id2relation[i] + "_rev"

        if 'use_self_loop' in args:
            self.use_self_loop = args['use_self_loop']
            if self.use_self_loop:
                id2relation[len(id2relation)] = "self_loop"

        self.id2relation = id2relation
        self.file_write = None
        self.device = device

    def evaluate(self, valid_data, test_batch_size=20, write_info=False):
        write_info = True
        self.model.eval()
        self.count = 0
        eps = self.eps
        id2entity = self.id2entity
        eval_loss, eval_acc, eval_max_acc = [], [], []
        f1s, hits, ems, precisions, recalls = [], [], [], [], []
        valid_data.reset_batches(is_sequential=True)
        num_epoch = math.ceil(valid_data.num_data / test_batch_size)
        if write_info and self.file_write is None:
            filename = os.path.join(self.args['checkpoint_dir'],
                                    "{}_test.info".format(self.args['experiment_name']))
            self.file_write = open(filename, "w")
        
        
        case_ct = {}
        max_local_entity = valid_data.max_local_entity
        ignore_prob = (1 - eps) / max_local_entity

        for iteration in tqdm(range(num_epoch)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0, test=True)
            with torch.no_grad():
                # forward 回傳五個值
                loss, raw_score, pred_dist, pred, tp_list = self.model(batch[:-1], training=False)

                # tp_list = [actions_per_step, h1, f1]
                if tp_list is not None:
                    actions_per_step, h1_list, f1_list = tp_list
                else:
                    actions_per_step, h1_list, f1_list = None, None, None

            if self.model_name == 'GraftNet':
                local_entity, query_entities, _, _, query_text, _, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch
            else:
                local_entity, query_entities, _, query_text, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch

            if write_info:
                obj_list = self.write_info(valid_data, actions_per_step, self.model.num_iter)

            candidate_entities = torch.from_numpy(local_entity).type('torch.LongTensor')
            true_answers = torch.from_numpy(answer_dist).type('torch.FloatTensor')
            query_entities = torch.from_numpy(query_entities).type('torch.LongTensor')
            eval_loss.append(loss.item())

            batch_size = pred_dist.size(0)
            batch_answers = answer_list
            batch_candidates = candidate_entities
            pad_ent_id = len(id2entity)

            for batch_id in range(batch_size):
                answers = batch_answers[batch_id]
                candidates = batch_candidates[batch_id, :].tolist()
                probs = pred_dist[batch_id, :].tolist()
                seed_entities = query_entities[batch_id, :].tolist()

                candidate2prob = []
                for c, p, s in zip(candidates, probs, seed_entities):
                    if s == 1.0:
                        continue
                    if c == pad_ent_id:
                        continue
                    if p < ignore_prob:
                        continue
                    candidate2prob.append((c, p))

                # 只取 top-1
                if len(candidate2prob) > 0:
                    top1_candidate2prob = [max(candidate2prob, key=lambda x: x[1])]
                else:
                    top1_candidate2prob = []

                precision, recall, f1, hit, em, case, retrived, ans = f1_and_hits(
                    answers, top1_candidate2prob, self.id2entity, self.entity2name, eps
                )


                if write_info:
                    tp_obj = obj_list[batch_id]
                    tp_obj['answers'] = ans
                    tp_obj['precison'] = precision
                    tp_obj['recall'] = recall
                    tp_obj['f1'] = f1
                    tp_obj['hit'] = hit
                    tp_obj['em'] = em
                    tp_obj['cand'] = retrived
                    self.file_write.write(json.dumps(tp_obj) + "\n")

                case_ct.setdefault(case, 0)
                case_ct[case] += 1
                f1s.append(f1)
                hits.append(hit)
                ems.append(em)
                precisions.append(precision)
                recalls.append(recall)

        print('evaluation.......')
        print('how many eval samples......', len(f1s))
        print('avg_em', np.mean(ems))
        print('avg_hits', np.mean(hits))
        print('avg_f1', np.mean(f1s))
        print('avg_precision', np.mean(precisions))
        print('avg_recall', np.mean(recalls))
        print(case_ct)

        if write_info:
            self.file_write.close()
            self.file_write = None
        
        return np.mean(f1s), np.mean(hits), np.mean(ems)


    def write_info(self, valid_data, actions_per_step, num_step):
        """
        將每個問題的推理步驟寫出來
        actions_per_step: List，每個 step 的動作 (tensor)，可能長度 < num_step
        """
        question_list = valid_data.get_quest()
        obj_list = [{} for _ in range(len(question_list))]

        for j in range(num_step):
            if actions_per_step is None or j >= len(actions_per_step):
                actions = None
            else:
                actions = actions_per_step[j].cpu().numpy()

            for i in range(len(question_list)):
                tp_obj = obj_list[i]
                q = question_list[i]
                tp_obj['question'] = q
                tp_obj[j] = {}

                if actions is not None:
                    action = actions[i]
                    rel_action = self.id2relation[action]
                    tp_obj[j]['rel_action'] = rel_action
                    tp_obj[j]['action'] = str(action)

        return obj_list
