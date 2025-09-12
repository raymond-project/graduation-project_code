import os
import time
import pandas as pd
import json
from openai import OpenAI
import re 

# 初始化 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# LLM 呼叫 (含 retry)
# ==============================
def call_openai_with_retry(prompt, idx, retries=3, timeout=60.0):
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f" 第 {idx+1} 筆，第 {i+1} 次失敗: {e}")
            if i == retries - 1:
                return '{"關鍵句子": "LLM失敗", "理由": "呼叫失敗"}'
            time.sleep(2 ** i)
    return '{"關鍵句子": "LLM失敗", "理由": "重試失敗"}'


def safe_json_parse(text, true_code):
    try:
        # 把可能的 ```json ... ``` 外殼去掉
        text = text.strip()
        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text)

        # 嘗試抓第一個 { ... } 區塊
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group())

        # 如果沒有 { }，回傳預設格式
        return {"原來答案": true_code, "關鍵句子": text, "理由": "非JSON輸出"}
    except Exception:
        return {"原來答案": true_code, "關鍵句子": "JSON解析失敗", "理由": "無"}
# ==============================
# 主程式：讀取錯誤檔案並更正
# ==============================
if __name__ == "__main__":
    input_file = r"/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換後/關鍵句錯誤.csv"
    output_file = r"/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換後/關鍵句更正.csv"

    df = pd.read_csv(input_file)

    corrected_results = []
    for idx, row in df.iterrows():
        true_code = str(row["原來答案"]).strip()
        pred_code = str(row["預測答案"]).strip()
        old_sentence = str(row["關鍵句子"])
        old_reason = str(row["理由"])
        report = str(row["reportData"])

        # === Prompt ===
        prompt = f"""
你是一個癌症登錄編碼助手。
現在有一份病理報告，並且已經知道人工登錄的正確編碼（原來答案）。
請根據「原來答案」以及病理報告，挑出能支持該答案的關鍵句子，並寫出理由。



編碼指引：
- 根據原發部位病理報告摘錄是否有神經侵襲。若無病理報告，則可以參考病歷記錄中有關病理資訊之描述。
- 可參考申報醫院與外院資料，優先摘錄神經侵襲之陽性資訊，若無外院資料則摘錄申報醫院資料。
- 病情惡化後所執行的病理報告則不採用。
- 病理報告若描述為無殘餘腫瘤(no residual tumor)時，應視為無神經侵襲。
- 任一原發部位病理報告記載有神經侵襲，編碼為1。
- 未接受前導性治療的個案，若有多份原發部位病理報告，且報告內容描述涵蓋有不確定、未侵襲或不詳等記錄時，應以未侵襲的記錄為優先，編碼為0。
- 所有原發部位病理報告皆未描述神經侵襲情形，編碼為7。
- 因病理樣本無法評估或判斷而未能描述神經侵襲侵犯情形，編碼為7。
- 個案為High grade dysplasia (severe dysplasia)或原位癌，編碼為8(不適用)。
- 原發部位未執行病理組織檢查或僅執行細胞學檢查、原發部位不詳(C80.9)、淋巴瘤、白血病、Plasma Cell Myeloma、GIST和NETs之個案與中樞神經系統之惡性腫瘤，應編碼為8 (不適用)。
- 若個案接受前導性治療，登錄原則如下：
  - 前導性治療執行前後任一原發部位病理報告記錄有神經侵襲，應編碼為1。
  - 若治療前後的病理報告皆描述未有神經侵襲時，應編碼為0。
  - 若治療前的病理報告描述為不確定是否有侵犯與未侵犯兩種記錄時，治療後有病理報告描述為未侵犯或無任何資訊請編碼7；治療後若無病理報告且無任何資訊編碼9。
  - 若治療前無任何資訊，治療後病理報告描述未侵犯或無資訊編碼9。

定義：
0 = 無神經侵襲
1 = 有神經侵襲
7 = 無法評估/未描述
8 = 不適用
9 = 病歷未記載或不詳

請輸出 JSON 格式：
{{
  "原來答案": "{true_code}",
  "關鍵句子": "...",
  "理由": "..."
}}

原來答案: {true_code}

病理報告：
{report}
"""


        llm_output = call_openai_with_retry(prompt, idx)
        parsed = safe_json_parse(llm_output, true_code)

        corrected_results.append({
            "原來答案": true_code,
            "更正後關鍵句子": parsed.get("關鍵句子", "未找到"),
            "更正後理由": parsed.get("理由", "無"),
            "reportData": report
        })

    # 輸出更正後 CSV
    out_df = pd.DataFrame(corrected_results)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"更正完成，共處理 {len(out_df)} 筆")
    print(f"輸出到 {output_file}")
