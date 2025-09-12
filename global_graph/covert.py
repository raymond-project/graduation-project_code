import pandas as pd
import os

# 讀取 CSV
csv_path = "/home/st426/system/global_graph/data_.csv"
df = pd.read_csv(csv_path)

# 指定輸出資料夾 (請改成你要的目錄)
output_dir = "/home/st426/system/global_graph/doc"

# 如果資料夾不存在就建立
os.makedirs(output_dir, exist_ok=True)

# 遍歷 reportData 欄位，逐筆輸出 txt
for idx, content in enumerate(df["reportData"], start=1):
    file_path = os.path.join(output_dir, f"report_{idx}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(content))

print(f"已完成！共輸出 {len(df)} 個 txt 檔到 {output_dir}")
