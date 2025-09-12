import pandas as pd

# 讀取資料
data_df = pd.read_csv('數據集(ALL)/data_.csv')
answer_df = pd.read_csv('數據集(ALL)/答案.csv')

# 擷取需要欄位
data_df = data_df[['cancerNo', 'reportData']]

answer_df = answer_df[['cancerNo', '組織類型']]

# 根據 cancerNo 合併
merged_df = pd.merge(data_df, answer_df, on='cancerNo', how='inner')

# 只保留 reportData 和 側性
result_df = merged_df[['組織類型', 'reportData']]

# 輸出成新 CSV
result_df.to_csv('reportData和組織類型.csv', index=False, encoding='utf-8-sig')
