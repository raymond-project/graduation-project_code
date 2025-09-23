
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
from evaluate import Evaluator



class Trainer_KBQA(object):                                         
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
            
            
            
                                        

        if model_name == 'ReaRev':                 
            self.model = ReaRev(self.args,  len(self.entity2id), self.num_kb_relation,
                                  self.num_word)

            
        
        
        
        
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

        dataset = load_data(args, tokenize)         
            
        
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
        eval_every = self.args['eval_every']
        print("Start Training------------------")

        all_con_losses = []
        train_h1s, train_f1s = [], []
        eval_h1s, eval_f1s = [], []

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()

            if self.decay_rate > 0:
                self.scheduler.step()

            
            all_con_losses.append(extras["contrastive"])

            train_h1 = np.mean(h1_list_all)
            train_f1 = np.mean(f1_list_all)
            train_h1s.append(train_h1)
            train_f1s.append(train_f1)

            self.logger.info(
                f"Epoch: {epoch+1}, total: {loss:.4f}, Contrastive: {extras['contrastive']:.4f}, "
                f"time: {time.time()-st:.2f}"
            )
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(train_h1, train_f1))

            # 每 eval_every 輪做 validation
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

        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

      
        plt.plot(range(1, len(all_con_losses)+1), all_con_losses, marker="^", label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curves")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()

        plt.plot(range(1, len(train_h1s)+1), train_h1s, marker="o", label="Train H1")
        plt.plot(range(1, len(train_f1s)+1), train_f1s, marker="x", label="Train F1")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Training H1/F1 Curve")
        plt.legend()
        plt.savefig("train_h1_f1_curve.png")
        plt.close()

        if eval_h1s and eval_f1s:
            plt.plot(np.arange(eval_every, (len(eval_h1s)+1)*eval_every, eval_every), eval_h1s, marker="o", label="Eval H1")
            plt.plot(np.arange(eval_every, (len(eval_f1s)+1)*eval_every, eval_every), eval_f1s, marker="x", label="Eval F1")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("Validation H1/F1 Curve")
            plt.legend()
            plt.savefig("eval_h1_f1_curve.png")
            plt.close()












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

    def train_epoch(self):
        self.model.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        h1_list_all, f1_list_all = [], []

        num_epoch = math.ceil(self.train_data.num_data / self.args['batch_size'])
        con_losses = []

        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            self.optim_model.zero_grad(set_to_none=True)

            loss, raw_score, pred_dist, pred, tp_list, loss_dict = self.model(batch, training=True)
            raw_score = torch.clamp(raw_score, -20.0, 20.0)

            tqdm.write(f"loss: {loss.item():.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get('gradient_clip', 0.5))
            self.optim_model.step()
            self.optim_model.zero_grad(set_to_none=True)

            losses.append(loss.item())
            con_losses.append(loss_dict["contrastive"])

            if tp_list is not None:
                h1_list, f1_list = tp_list
                h1_list_all.extend(h1_list)
                f1_list_all.extend(f1_list)

        extras = {"contrastive": np.mean(con_losses) if con_losses else 0.0}
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
        model.load_state_dict(model_state_dict, strict=False)

