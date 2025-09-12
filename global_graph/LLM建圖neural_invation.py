import os
import re
import time
import json
import pandas as pd
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def safe_json_parse(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print(" 模型輸出非 JSON：", s[:200])
            return []

def log_error(idx, report_text, error):
    with open("error_log.txt", "a", encoding="utf-8") as logf:
        logf.write(f" 第 {idx+1} 筆失敗: {error}\n")
        logf.write(f"報告內容: {report_text[:200]}...\n\n")

def call_openai_with_retry(prompt, idx, report_text, retries=3, timeout=60.0):
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout
            )
        except Exception as e:
            print(f" 第 {idx+1} 筆，第 {i+1} 次失敗: {e}")
            if i == retries - 1:
                log_error(idx, report_text, str(e))
            time.sleep(2 ** i)
    return None

# ==============================
# 載入規則 (從 CSV)
# ==============================

def load_rules_from_csv(csv_path):
    df_rules = pd.read_csv(csv_path)
    rules_text = "【神經侵犯編碼規則】\n"
    for _, row in df_rules.iterrows():
        rules_text += f"- {row['編碼']}: {row['定義神經侵襲編碼規則']}\n"
    return rules_text

# ==============================
# 主要函式
# ==============================

def build_single_chain(report_text, evidence_sentence, correct_code, idx=0):
    rules = load_rules_from_csv(rules_path)

    prompt = f"""
You are a cancer registry reasoning assistant. 
Given a pathology report and a key evidence sentence, extract a minimal ordered reasoning path 
that leads to the coding decision.

{rules}

【Rules for Output】
1. Always end the chain with the correct code: {correct_code}
2. Nodes must appear in a logical order from evidence_sentence tokens, plus any additional decisive context 
   from the full report (reportData).
3. Do not branch. Output a single linear chain.
4. Keep each node short (1–5 words).
5. Output JSON array only.

【Input】
reportData:
{report_text}

evidence_sentence:
{evidence_sentence}

Output:
Return only a valid JSON array of nodes. 
Do not include any extra text, explanation, or formatting.
"""

    resp = call_openai_with_retry(prompt, idx=idx, report_text=report_text)
    if resp is None:
        return []
    raw = resp.choices[0].message.content.strip()
    nodes = safe_json_parse(raw)

    triples = []
    for i in range(len(nodes) - 1):
        triples.append({
            "report_id": idx,
            "subject": nodes[i],
            "relation": "leads_to",
            "object": nodes[i + 1],
            "evidence_sentence": evidence_sentence
        })

    if triples:
        triples[-1]["object"] = str(correct_code)

    return triples

# ==============================
# 主程式
# ==============================

if __name__ == "__main__":
    # 讀取你給的資料檔案
    df = pd.read_csv(r"/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換前/reportData和神經侵襲.csv")

    big_graph = []
    output_file = "neural_invasion_graph.json"

    for idx, row in df.iterrows():
        report_text = row["reportData"]
        correct_code = str(row["神經侵襲"]).strip()   # 直接取正確答案
        evidence_sentence = row["reportData"][:100]   # 如果沒有 sentence 欄，就先用報告前100字當 evidence

        triples = build_single_chain(report_text, evidence_sentence, correct_code, idx)
        big_graph.extend(triples)

        # 即時更新大圖 JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(big_graph, f, indent=2, ensure_ascii=False)

        print(f" 已處理第 {idx+1} 筆, 累積三元組 {len(big_graph)} 條 (已即時寫入 {output_file})")

    print(f" 大圖已完成，輸出到 {output_file}")
    print(" 若有失敗，請查看 error_log.txt")
