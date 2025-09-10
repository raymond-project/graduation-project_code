import json
import random
from collections import defaultdict

# 檔案路徑
input_file = "/home/st426/system/GNN-RAG/gnn/data/surdis3/gnn_train.jsonl"
train_file = "/home/st426/system/GNN-RAG/gnn/data/surdis3/train.json"
dev_file   = "/home/st426/system/GNN-RAG/gnn/data/surdis3/dev.json"
test_file  = "/home/st426/system/GNN-RAG/gnn/data/surdis3/test.json"

random.seed(42)  # 固定隨機種子，避免每次切分不同

# 讀取並轉換 kb_id 為字串
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if "answers" in obj:
            for ans in obj["answers"]:
                if "kb_id" in ans:
                    ans["kb_id"] = str(ans["kb_id"])
        data.append(obj)

# 覆寫回原始 jsonl (每行一筆 JSON)
with open(input_file, "w", encoding="utf-8") as f:
    for obj in data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 依據 kb_id 分組
groups = defaultdict(list)
for item in data:
    kb_id = item["answers"][0]["kb_id"]
    groups[kb_id].append(item)

train, dev, test = [], [], []

# 每組 kb_id 內切分
for kb_id, items in groups.items():
    if len(items) == 1:
        train.extend(items)
    else:
        for item in items:
            r = random.random()
            if r < 0.8:
                train.append(item)
            elif r < 0.9:
                dev.append(item)
            else:
                test.append(item)

# 儲存為「每行一個 JSON object」，副檔名為 .json
def save_as_json_lines(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for obj in dataset:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

save_as_json_lines(train_file, train)
save_as_json_lines(dev_file, dev)
save_as_json_lines(test_file, test)

print("處理完成！")
for k, v in groups.items():
    print(f"kb_id={k} → 共 {len(v)} 筆 (train={sum(1 for x in train if x['answers'][0]['kb_id']==k)}, dev={sum(1 for x in dev if x['answers'][0]['kb_id']==k)}, test={sum(1 for x in test if x['answers'][0]['kb_id']==k)})")

print(f"總數: {len(data)} 筆, train={len(train)}, dev={len(dev)}, test={len(test)}")
