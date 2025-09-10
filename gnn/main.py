import argparse

from utils import create_logger
import torch
import numpy as np
import os
import time






#其他的py  parsing 記得要分開寫 這樣好調整參數
#from Models.ReaRev.rearev import 
from train_model import GraphModelTrainer
from parsing import add_parse_args

parser = argparse.ArgumentParser()
add_parse_args(parser)
args = parser.parse_args()




args.use_cuda = torch.cuda.is_available()





np.random.seed(args.seed)
torch.manual_seed(args.seed)





#資料及若沒給參數=> 指定這次訓練或評估實驗的名稱  我把{}-{}-{}  分別設為 資料集  model_name 跟 時間
if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )



#開跑 紀錄 => 訓練\評估
def main():
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    
    
    
    trainer = GraphModelTrainer(args=vars(args), model_name=args.model_name, logger=logger)
    
    
    

    
    #這邊我用設定訓練模式或評估模式  讓我的sh 可以用
    #訓練模式
    if not args.is_eval:                      
        trainer.train(0, args.num_epoch - 1)
        
        
    #評估模式    
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("載入預訓練模型 {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)     #需載入 .ckpt  


if __name__ == '__main__':
    main() 
