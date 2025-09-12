import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案（請依照你的實際檔案名稱修改）
df = pd.read_csv("/home/st426/system/GNN-RAG/數據集(ALL)/組織型態_前/reportData和組織類型.csv")

# 檢查欄位名稱正確
assert "組織類型" in df.columns, "缺少 '組織類型' 欄位"

# 計算數據分布
value_counts = df["組織類型"].value_counts()

# 繪圖
fig, ax = plt.subplots()
value_counts.plot(kind='bar', ax=ax)
ax.set_title("tissueTypeRules")
ax.set_xlabel("Label")
ax.set_ylabel("Num")
plt.xticks(rotation=0)

# 儲存為 PNG 圖片
plt.savefig("tissueTypeRules.csv.png")
