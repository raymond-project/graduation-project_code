訓練=>
python main.py ReaRev   --entity_dim 150   --num_epoch 10   --batch_size 8   --eval_every 2   --data_folder data/surdis3/   --lm ClinicalBERT   --num_iter 4   --num_ins 5   --num_gnn 6   --relation_word_emb True


"""
訓練 ReaRev 的 GNN 模型，模型會使用 BERT（實際上是 BioBERT），然後從 data/.../ 裡面抓資料來跑 100 個 epoch，期間
每 2 epoch 驗證一次，整個訓練過程會依據你給的參數來調整 GNN 層、instruction 步驟等設定，並把訓練過程以 biobert-experiment為名稱儲存。


把 --name  改成你的任務名稱

把 --data_folder 改成你自己的資料夾路徑

調整 --num_epoch, --entity_dim, --batch_size 根據你 GPU 和資料量來調 


"""


使用方式=>
#訓練完的gnn
python main.py ReaRev    --entity_dim 150   --data_folder data/surdis3/   --lm ClinicalBERT   --lm_frozen 1   --num_iter 2   --num_ins 2   --num_gnn 6   --relation_word_emb True   --load_experiment /home/st426/system/GNN-RAG/gnn/checkpoint/pretrain/-final.ckpt   --is_eval    --name cancer-scibert
  
