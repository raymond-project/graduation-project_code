import pandas as pd

# 檔案路徑
file_path = "/home/st426/system/global_graph/預測結果_錯誤_surdis.csv"

# 讀取 CSV
df = pd.read_csv(file_path)

# 確保欄位轉成數字 (避免有空白或非數字內容)
df["原發部位手術切緣距離"] = pd.to_numeric(df["原發部位手術切緣距離"], errors="coerce")

# 過濾掉 int = 999 的整筆資料
df = df[df["原發部位手術切緣距離"] != 999]

# 存回原檔 (覆寫)
df.to_csv(file_path, index=False, encoding="utf-8-sig")

print("✅ 已移除原發部位手術切緣距離 = 999 的資料，並覆寫檔案。")
