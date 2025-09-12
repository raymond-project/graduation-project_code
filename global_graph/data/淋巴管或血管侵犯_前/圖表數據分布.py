import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案（請依照你的實際檔案名稱修改）
df = pd.read_csv("/home/st426/system/GNN-RAG/數據集(ALL)/淋巴管或血管侵犯_前/reportData和淋巴管或血管侵犯.csv")

# 檢查欄位名稱正確
assert "淋巴管或血管侵犯" in df.columns, "缺少 '淋巴管或血管侵犯' 欄位"

# 計算數據分布
value_counts = df["淋巴管或血管侵犯"].value_counts()

# 繪圖
fig, ax = plt.subplots()
value_counts.plot(kind='bar', ax=ax)
ax.set_title("Lymphovascular_invasion")
ax.set_xlabel("Label")
ax.set_ylabel("Num")
plt.xticks(rotation=0)

# 儲存為 PNG 圖片
plt.savefig("Lymphovascular_invasion.png")
