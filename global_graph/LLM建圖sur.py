import os
import re
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI

# 初始化 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# 工具函式
# ==============================

def safe_json_parse(s: str):
    """保險的 JSON parser，避免模型輸出不乾淨"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print("⚠️ 模型輸出非 JSON：", s)
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
# 新版編碼規則
# ==============================

final_codes = {"0","1","2","3","4","5","7","8","9","A","B","C","D","E","F"}

Rule = """
編碼,定義原發部位手術邊緣編碼規則
0,手術紀錄描述無殘存腫瘤。病理報告描述「Uninvolved」。
1,只知道有殘存的侵襲性癌細胞，至於其他更詳細的情形則不清楚，手術紀錄描述無殘存腫瘤。
2,病理報告中巨觀無殘存腫瘤，且手術紀錄描述無殘存腫瘤，僅在顯微鏡下看到殘存的侵襲性癌細胞。病理報告描述「involved」。
3,病理報告描述在肉眼下就可以看到殘存的侵襲性癌細胞，顯微鏡下及手術紀錄皆無描述手術邊緣狀態。
4,病理報告在肉眼及顯微鏡下皆看到殘存的侵襲性癌細胞，手術紀錄描述無殘存腫瘤。
5,病理報告描述侵襲癌手術邊緣狀態為very close或may not be free。侵襲癌病理報告僅描述<1mm且未明示手術邊緣狀態。
7,病理報告描述手術邊緣狀態無法評估。
8,未針對原發腫瘤部位進行手術。手術方式編碼為10-19者。攝護腺癌個案僅接受TURP。
A,手術紀錄描述有殘存腫瘤，或為腫瘤部份切除(R2 resection)，但病理報告描述無殘存腫瘤、無法評估或不清楚。
B,手術紀錄描述有殘存腫瘤，或為腫瘤部份切除(R2 resection)，同時病理報告描述亦有殘存侵襲性癌細胞。
C,病理報告描述手術邊緣為 high grade、moderate dysplasia、severe dysplasia、carcinoma in situ。
D,病理報告描述手術邊緣為 mild dysplasia or low grade。
E,病理報告描述手術邊緣為 dysplasia，未明示為 high or low grade。
F,病理報告描述原位癌/分化不良手術邊緣狀態為 very close 或 may not be free。病理報告僅描述 <1mm 且未明示手術邊緣狀態。
9,不知道個案是否有接受原發部位手術。原發部位為淋巴結的淋巴癌、原發不明或病歷未記載。
"""

# ==============================
# Triple Builder
# ==============================

def connect_isolated_nodes(triples, correct_code, evidence_sentence="__bridge__"):
    """避免孤立節點，強制接到正確 code"""
    G = nx.DiGraph()
    for t in triples:
        G.add_edge(t["subject"], t["object"])

    connected = set()
    for node in G.nodes:
        if correct_code in G.nodes and nx.has_path(G, node, correct_code):
            connected.add(node)

    isolated = [n for n in G.nodes if n not in connected and n not in final_codes]

    for node in isolated:
        triples.append({
            "subject": node,
            "relation": "leads_to",
            "object": correct_code,
            "evidence_sentence": evidence_sentence
        })
    return triples

def build_graph_from_labeled(report_text, correct_code, Rule, idx=0):
    """呼叫 GPT 生成三元組"""
    prompt = f"""
你是一個癌症登記知識圖譜建構助手。  
輸入有：(1) 病理報告全文 (reportData)，(2) 已知的正確終端編碼 (correct_code)。  
任務：輸出 JSON 陣列，每筆是 (subject, relation, object)。

【終端編碼集合】  
只能使用這些代碼: ["0","1","2","3","4","5","7","8","9","A","B","C","D","E","F"]  

{Rule}

【規則】  
1. subject/object 必須來自 reportData 的片段 (trim 可) 或正確 code，不可捏造。  
2. relation 只能選擇: ["status","associated_with","location","measured_in","type_of","evaluation","implies_code","leads_to","corresponds_to"]  
3. 至少有一條推理鏈，最後必須到正確 code "{correct_code}"。  
4. 請確保沒有孤立節點。  
5. 輸出必須是純 JSON 陣列。  

【輸入】  
reportData:  
{report_text}
"""
    response = call_openai_with_retry(prompt, idx, report_text)
    if response is None:
        return []

    raw_output = response.choices[0].message.content.strip()
    triples = safe_json_parse(raw_output)
    triples = connect_isolated_nodes(triples, correct_code)
    return triples


# ==============================
# 主程式
# ==============================

if __name__ == "__main__":
    INPUT_CSV = "/home/st426/system/global_graph/data/sur/reportData和原發部位手術邊緣.csv"
    OUTPUT_DIR = "/home/st426/system/global_graph/graph_sur"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    big_graph = []
    output_file = os.path.join(OUTPUT_DIR, "surgical_margin_graph.json")

    for idx, row in df.iterrows():
        report_text = str(row["reportData"])
        correct_code = str(row["原發部位手術邊緣"]).strip()

        triples = build_graph_from_labeled(report_text, correct_code, Rule, idx)
        big_graph.extend(triples)

        # 更新 JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(big_graph, f, indent=2, ensure_ascii=False)


        print(f"✅ 第 {idx+1} 筆完成, 累積 {len(big_graph)} 條")

    print(f"🎉 大圖完成 → {output_file}")
    print("📄 錯誤請查看 error_log.txt")
