import os
import pandas as pd
import json
from openai import OpenAI

# 初始化 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 編碼規則 (放在 prompt)
INSTRUCTION = """
You are a pathology report coding assistant.
Task: From the pathology report, determine lymph-vascular invasion (LVI) code.

Output format (JSON only):
{
  "reason": "<the extracted sentence or NA>",
  "final_code": "<0/1/7/8/9>"
}

Coding rules:
0 無淋巴管或血管侵犯。病理報告描述 "Not identified", "absent" 或 "no residual tumor"。
1 有淋巴管或血管侵犯。病理報告描述 "present"。同義詞包括：
   Angiolymphatic invasion, Blood vessel invasion, Lymph-vascular emboli,
   Lymphatic invasion, Lymphvascular invasion, Vascular invasion。
7 病理報告描述為 NA 或無法評估。病理樣本非常小，無法判斷。病理報告提及樣本不足以判斷是否有淋巴管或血管侵犯。所有原發部位病理報告皆未描述淋巴管或血管侵犯情形。原發部位有執行病理檢查，但病理報告描述無惡性細胞。或治療後病理報告僅記錄「不確定」。
8 不適用。有下列情況之一表示不適用侵犯規則：GIST、NETs、High grade dysplasia (Severe dysplasia)、原位癌 (in situ)、淋巴瘤、白血病、Plasma cell myeloma、中樞神經系統惡性腫瘤、原發部位不詳。或原發部位未執行病理檢查、僅執行細胞學檢查。病情惡化後的病理報告亦不採用。
9 病歷或病理報告未記載或不詳。若接受前導性治療但前後病理報告皆缺乏資訊時，也編碼為9。
"""

# 輸入檔案
input_path = "/home/st426/system/global_graph/data/淋巴管或血管侵犯_前/reportData和淋巴管或血管侵犯.csv"
df = pd.read_csv(input_path)

results = []

for idx, row in df.iterrows():
    report = str(row["reportData"]).strip()
    true_code = str(row["淋巴管或血管侵犯"]).strip()

    prompt = INSTRUCTION + f"\n\nReport:\n{report}"


    resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    response_format={"type": "json_object"}  
)

    parsed = json.loads(resp.choices[0].message.content.strip())
    pred_code = parsed.get("final_code", "9")
    reason = parsed.get("reason", "NA")


    results.append({
        "原來答案": true_code,
        "預測答案": pred_code,
        "那句分類句子": reason,
        "reportData": report
    })

# 轉成 DataFrame
out_df = pd.DataFrame(results)

# 輸出 CSV
output_path = "/home/st426/system/global_graph/data/淋巴管或血管侵犯_前/prediction.csv"
out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"已完成，輸出檔案在: {output_path}")
