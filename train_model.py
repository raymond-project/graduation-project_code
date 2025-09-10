
from utils import create_logger
import time
import numpy as np
import os, math
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
import random

from tqdm import tqdm
tqdm.monitor_iterval = 0



from dataset_load import load_data
from dataset_load_graft import load_data_graft
from models.ReaRev.rearev import ReaRev
from models.NSM.nsm import NSM
from models.GraftNet.graftnet import GraftNet
from evaluate import Evaluator

class GraphModelTrainer(object):                                         
    def __init__(self, args, model_name, logger=None):
        #print('Trainer here')
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.best_h1b = 0.0
        self.best_f1b = 0.0
        self.eps = args['eps']
        self.warmup_epoch = args['warmup_epoch']
        self.learning_rate = self.args['lr']
        self.test_batch_size = args['test_batch_size']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.reset_time = 0
        self.load_data(args, args['lm'])
        


        if 'decay_rate' in args:
            self.decay_rate = args['decay_rate']
        else:
            self.decay_rate = 0.98               #預設給0.98 decay_rate 如果沒寫
            
            
            
                                                # 底下都是一些gnn model 

        if model_name == 'ReaRev':                 
            self.model = ReaRev(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        elif model_name == 'NSM':
            self.model = NSM(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        elif model_name == 'GraftNet':
            self.model = GraftNet(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
            
        """ 
        elif model_name == 'NuTrea':
            self.model = NuTrea(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)
        """    
            
        
        
        
        
        # 將關係（relation）的自然語言描述轉換成嵌入向量，並傳給 GNN 模型用來幫助推理
        if args['relation_word_emb']:
            #self.model.use_rel_texts(self.rel_texts, self.rel_texts_inv)
            self.model.encode_rel_texts(self.rel_texts, self.rel_texts_inv) # 導向 =>


        self.model.to(self.device) 
        self.evaluator = Evaluator(args=args, model=self.model, entity2id=self.entity2id,
                                       relation2id=self.relation2id, device=self.device)  #為gnn model 打分
        self.load_pretrain()   # load_pretrain 的 def funtion 來自args
        self.optim_def()       #優化器（Adam）與學習率 scheduler（ExponentialLR），自動篩選需要訓練的參數、設學習率

        
        self.num_relation =  self.num_kb_relation #圖譜中總共有幾種關係
        self.num_entity = len(self.entity2id)     #多少節點
        self.num_word = len(self.word2id)         #總詞彙數                             
        print("Entity: {}, Relation: {}, Word: {}".format(self.num_entity, self.num_relation, self.num_word))


        #args 中的參數   遍歷 args 字典中的每一個鍵值對  #????
        for k, v in args.items():  
            if k.endswith('dim'): 
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

    def optim_def(self):
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim_model = optim.Adam(trainable, lr=self.learning_rate, weight_decay=1e-4)  # ← 新增 weight_decay
        if self.decay_rate > 0:
            self.scheduler = ExponentialLR(self.optim_model, self.decay_rate)

            
            





    def load_data(self, args, tokenize):                                
        if args["model_name"] == "GraftNet":
            dataset = load_data_graft(args, tokenize)   # from dataset_load_graft import load_data_graft  #load_data_graft(...) 是專門 用在GraftNet 
        else:
            dataset = load_data(args, tokenize)         #from dataset_load import load_data   #ReaRev、NSM 用的是 load_data(...)
            
        
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        
        #
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]

        #
        self.num_word = dataset["num_word"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)
        
        #
        self.rel_texts = dataset["rel_texts"]
        self.rel_texts_inv = dataset["rel_texts_inv"]



    def load_pretrain(self):
        args = self.args  #有訓練過的模型
        
        
        
        if args['load_experiment'] is not None:  #從頭開始訓練，不需要載入預訓練模型
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)







    # data 來源 self.evaluate(self.valid_data  
    #write_info  =>  是否寫出 .info 給下游 RAG 用 
    def evaluate(self, data, test_batch_size=20, write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, write_info)










    def train(self, start_epoch, end_epoch):
        eval_every = self.args['eval_every']  # 每多少個 epoch 驗證一次
        print("Start Training------------------")

        # ====== 紀錄用 ======
        all_losses = []
        train_h1s, train_f1s = [], []
        eval_h1s, eval_f1s = [], []

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()

            if self.decay_rate > 0:
                self.scheduler.step()

            # ====== 紀錄 loss 與 training 指標 ======
            all_losses.append(loss)
            train_h1 = np.mean(h1_list_all)
            train_f1 = np.mean(f1_list_all)
            train_h1s.append(train_h1)
            train_f1s.append(train_f1)

            self.logger.info("Epoch: {}, loss : {:.4f}, time: {:.2f}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(train_h1, train_f1))
            
            
            rand_idx = random.randint(0, len(self.train_data.data) - 1)
            sample = self.train_data.data[rand_idx]
            sentence = sample["question"]
            tokens = self.train_data.tokenizer.encode_plus(
                    text=sentence,
                    max_length=self.train_data.max_query_word,
                    padding='max_length',
                    return_attention_mask=False,
                    truncation=True
                )
            q_input = torch.tensor(tokens['input_ids']).unsqueeze(0).to(self.device)
            with torch.no_grad():
                instr_list, _ = self.model.instruction(q_input)
                # instr_list 是多 step instruction 的 list
                emb = instr_list[0].cpu().numpy()  # 取第一個 instruction
            print(f"[Epoch {epoch+1}] 句子: {sentence}")
            print(f"   embedding 前10維: {emb[0][:10]}")

            # ====== 每 eval_every 輪驗證 ======
            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1, eval_em = self.evaluate(self.valid_data, self.test_batch_size)
                eval_h1s.append(eval_h1)
                eval_f1s.append(eval_f1)

                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

                do_test = False
                if epoch > self.warmup_epoch:
                    if eval_h1 > self.best_h1:
                        self.best_h1 = eval_h1
                        self.save_ckpt("h1")
                        self.logger.info("BEST EVAL H1: {:.4f}".format(eval_h1))
                        do_test = True
                    if eval_f1 > self.best_f1:
                        self.best_f1 = eval_f1
                        self.save_ckpt("f1")
                        self.logger.info("BEST EVAL F1: {:.4f}".format(eval_f1))
                        do_test = True

                eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size)
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

        # ====== 訓練結束，儲存模型 ======
        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

        # ====== 繪圖 ======
        # Loss curve
        plt.plot(range(1, len(all_losses)+1), all_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.savefig("loss_curve.png")
        plt.close()

        # Training H1/F1 curve
        plt.plot(range(1, len(train_h1s)+1), train_h1s, marker="o", label="Train H1")
        plt.plot(range(1, len(train_f1s)+1), train_f1s, marker="x", label="Train F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Training H1/F1 Curve")
        plt.legend()
        plt.savefig("train_h1_f1_curve.png")
        plt.close()

        # Eval H1/F1 curve
        if eval_h1s and eval_f1s:  # 確保有資料
            plt.plot(np.arange(eval_every, (len(eval_h1s)+1)*eval_every, eval_every), eval_h1s, marker="o", label="Eval H1")
            plt.plot(np.arange(eval_every, (len(eval_f1s)+1)*eval_every, eval_every), eval_f1s, marker="x", label="Eval F1")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("Validation H1/F1 Curve")
            plt.legend()
            plt.savefig("eval_h1_f1_curve.png")
            plt.close()
         















######## 下面還沒看
######## 下面還沒看
######## 下面還沒看
######## 下面還沒看


    def evaluate_best(self):
        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size,  write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size, write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        eval_f1, eval_hits, eval_ems = self.evaluate(self.valid_data, self.test_batch_size, write_info=False)
        self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_hits, eval_ems))
        test_f1, test_hits, test_ems = self.evaluate(self.test_data, self.test_batch_size, write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(test_f1, test_hits, test_ems))
    def debug_tensor(self,name, t, iteration):
        if t is None:
            print(f"[DEBUG][iter={iteration}] {name}: None")
            return
        t = t.detach()
        nan = torch.isnan(t).any().item()
        print(
            f"[DEBUG][iter={iteration}] {name}: "
            f"shape={tuple(t.shape)} "
            f"min={t.min().item():.6f} "
            f"max={t.max().item():.6f} "
            f"mean={t.mean().item():.6f} "
            f"std={t.std().item():.6f} "
            f"norm={t.norm().item():.6f} "
            f"has_nan={nan}"
        )

    def train_epoch(self):
        self.model.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        h1_list_all, f1_list_all = [], []

        num_epoch = math.ceil(self.train_data.num_data / self.args['batch_size'])

        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list) = batch[2]
            print(f"[DEBUG] iter={iteration} | heads={len(batch_heads)} | rels={len(batch_rels)} | "
                f"tails={len(batch_tails)} | fact_ids={len(fact_ids)}")

            # ---- forward
            self.optim_model.zero_grad(set_to_none=True)

            # 改成用 raw_score
            loss, raw_score, pred_dist, pred, tp_list = self.model(batch, training=True)

            # raw_score = float logits (可正可負)
            # pred_dist = 機率分布 (0~1)
            # pred = argmax (long)

            print(f"[DEBUG][iter={iteration}] raw_score stats | "
                f"min={raw_score.min().item():.6f}, "
                f"max={raw_score.max().item():.6f}, "
                f"mean={raw_score.mean().item():.6f}, "
                f"std={raw_score.std().item():.6f}")

            # clamp 保護，避免 loss overflow
            raw_score = torch.clamp(raw_score, -20.0, 20.0)

            tqdm.write(f"loss: {loss.item():.4f}")

            # (1) 檢查 loss
            if not torch.isfinite(loss):
                msg = f"[FATAL] Non-finite loss at iter={iteration}: {loss.item()}"
                try:
                    self.logger.error(msg)
                except Exception:
                    print(msg)
                raise RuntimeError(msg)

            # ======== [debug kb_self_linear 前向] ========
            for name, module in self.model.named_modules():
                if "kb_self_linear" in name:
                    if hasattr(module, "input_debug"):
                        self.debug_tensor(f"{name}.input(before backward)", module.input_debug, iteration)
                    if hasattr(module, "output_debug"):
                        self.debug_tensor(f"{name}.output(before backward)", module.output_debug, iteration)
                    if hasattr(module, "weight"):
                        self.debug_tensor(f"{name}.weight", module.weight.data, iteration)

            # ---- backward
            loss.backward()

            # ======== [debug kb_self_linear 反向] ========
            for pname, p in self.model.named_parameters():
                if "kb_self_linear" in pname:
                    if p.grad is not None:
                        self.debug_tensor(f"{pname}.grad(after backward)", p.grad, iteration)

            # (2) 檢查 gradient
            bad_grad_name = None
            for name, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad_name = name
                    break
            if bad_grad_name is not None:
                msg = f"[FATAL] Non-finite gradient at iter={iteration} in param: {bad_grad_name}"
                try:
                    self.logger.error(msg)
                except Exception:
                    print(msg)
                raise RuntimeError(msg)

            # 可選：梯度裁切
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.get('gradient_clip', 0.5)
            )
            if not torch.isfinite(total_norm):
                msg = f"[FATAL] Non-finite grad-norm after clip at iter={iteration}"
                try:
                    self.logger.error(msg)
                except Exception:
                    print(msg)
                raise RuntimeError(msg)

            # ---- update
            self.optim_model.step()
            self.optim_model.zero_grad(set_to_none=True)

            losses.append(loss.item())

            # 收集 metric
            if tp_list is not None:
                h1_list, f1_list = tp_list
                h1_list_all.extend(h1_list)
                f1_list_all.extend(f1_list)

        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all





    
    def save_ckpt(self, reason="h1"):
        model = self.model
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        print("Best %s, save model as %s" %(reason, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]

        model = self.model
        #self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        model.load_state_dict(model_state_dict, strict=False)

