import os
import re
import time
import json
import pandas as pd
from openai import OpenAI

# 初始化 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FINAL_CODES = {"000","001~979","980","987","988","990","991"}

def normalize_node(node: str) -> str:
    if not node:
        return node
    node = node.strip().lower()
    if node.startswith("code "):
        node = node.replace("code ", "")
    if node.startswith("code:"):
        node = node.replace("code:", "")
    node = node.strip()
    # 嘗試轉換成合法終端碼
    node = normalize_final_code(node)
    return node


def safe_json_parse(s: str):
    """保險的 JSON parser，避免模型輸出不乾淨"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print("⚠️ 模型輸出非 JSON：", s[:200])
            return []

def log_error(idx, report_text, error):
    """紀錄失敗案例"""
    with open("error_log.txt", "a", encoding="utf-8") as logf:
        logf.write(f"❌ 第 {idx+1} 筆失敗: {error}\n")
        logf.write(f"報告內容: {report_text[:200]}...\n\n")

def call_openai_with_retry(prompt, idx, report_text, retries=3, timeout=60.0):
    """呼叫 GPT，內建 retry 機制"""
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout
            )
        except Exception as e:
            print(f"⚠️ 第 {idx+1} 筆，第 {i+1} 次失敗: {e}")
            if i == retries - 1:
                log_error(idx, report_text, str(e))
            time.sleep(2 ** i)
    return None

# ==============================
# 規則
# ==============================

def build_single_chain(report_text, evidence_sentence, correct_code, idx=0):
    rules = """
    【Surgical Margin Distance Coding Rules】
    - 000: 手術切緣陽性。病理報告描述「involved」。小於1mm，手術切緣明示為陽性。
    - 001~979: 手術切緣狀態為陰性，則記錄實際手術切緣距離，以0.1mm為單位。如:10 mm=100, 0.1 mm=001。
    - 980: 手術切緣距離大於98mm。
    - 987: 僅描述 very close、may not be free，且未描述切緣距離。
    - 988: 不適用。未執行原發腫瘤部位手術，或病理報告註明無法評估。
    - 990: 再切除後無殘餘腫瘤，或前導性治療後手術標本顯示無殘餘腫瘤。
    - 991: 手術邊緣為非侵襲癌 (原位癌或殘存分化不良/異型增生)。
    """

    prompt = f"""
You are a cancer registry reasoning assistant. 
Given a pathology report and a key evidence sentence, extract a minimal ordered reasoning path 
that leads to the coding decision.

{rules}

【Rules for Output】
1. Always end the chain with the correct code: {correct_code}
2. If the raw report contains a distance (e.g., "60 mm"), map it to the umbrella code "001~979".
3. If the report matches special rules (TURP → 988, no residual tumor → 990, CIS/dysplasia → 991),
   include that context before reaching the code.
4. Nodes must appear in a logical order from evidence_sentence tokens, plus any additional decisive context 
   from the full report (reportData).
5. Do not branch. Output a single linear chain.
6. Keep each node short (1–5 words).
7. Output JSON array only.

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
        subj = normalize_node(nodes[i])
        obj = normalize_node(nodes[i + 1])
        triples.append({
            "report_id": idx,
            "subject": subj,
            "relation": "leads_to",
            "object": obj,
            "evidence_sentence": evidence_sentence
        })

    
    if triples:
        triples[-1]["object"] = correct_code

    return triples

# ==============================
# Code Normalizer
# ==============================

def normalize_final_code(val: str) -> str:
    val = val.strip()
    if val in {"000","001~979","980","987","988","990","991"}:
        return val
    try:
        num = int(val)
        if 1 <= num <= 979:
            return "001~979"
        elif num == 0:
            return "000"
        elif num >= 980:
            return "980"
    except:
        pass
    return val


# ==============================
# 主程式
# ==============================

if __name__ == "__main__":
    df = pd.read_csv(r"/home/st426/system/global_graph/預測結果_正確_surdis.csv")

    big_graph = []
    output_file = "surgical_margin_graph.json"

    for idx, row in df.iterrows():
        report_text = row["reportData"]
        raw_code = str(row["原發部位手術切緣距離"]).strip()
        correct_code = normalize_final_code(raw_code)
        evidence_sentence = row["sentence"]

        triples = build_single_chain(report_text, evidence_sentence, correct_code, idx)
        big_graph.extend(triples)

        # 即時更新大圖 JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(big_graph, f, indent=2, ensure_ascii=False)

        print(f"✅ 已處理第 {idx+1} 筆, 累積三元組 {len(big_graph)} 條 (已即時寫入 {output_file})")

    print(f"🎉 大圖已完成，輸出到 {output_file}")
    print("📄 若有失敗，請查看 error_log.txt")
