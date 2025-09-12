import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案（請依照你的實際檔案名稱修改）
df = pd.read_csv("/home/st426/system/GNN-RAG/數據集(ALL)/Neural_Invasion_Detection轉換前/reportData和神經侵襲.csv")

# 檢查欄位名稱正確
assert "神經侵襲" in df.columns, "缺少 '神經侵襲' 欄位"

# 計算數據分布
value_counts = df["神經侵襲"].value_counts()

# 繪圖
fig, ax = plt.subplots()
value_counts.plot(kind='bar', ax=ax)
ax.set_title("Neural_Invasion_Detection")
ax.set_xlabel("Label")
ax.set_ylabel("Num")
plt.xticks(rotation=0)

# 儲存為 PNG 圖片
plt.savefig("neural_invasion_distribution.png")
