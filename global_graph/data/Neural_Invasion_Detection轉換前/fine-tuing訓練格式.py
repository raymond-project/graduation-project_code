import pandas as pd
import json
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source="zh-TW", target="en")

INSTRUCTION = """You are a pathology report coding assistant.
Task: Select one sentence from the pathology report that states perineural invasion.
If none, return "NONE".

Output format (JSON only):
{
  "reason": "<short reason, one sentence only>",
  "final_code": "<0/1/7/8/9>"
}

Codes:
0 = Not identified
1 = Present
7 = Not mentioned / not evaluable
8 = Not applicable (CIS / lymphoma / CNS tumor / etc.)
9 = Unknown
""" 

def translate_text(text):
    text = str(text).strip()
    return translator.translate(text)

correct_path = "/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換前/關鍵句正確.csv"
fix_path = "/home/st426/system/global_graph/data/Neural_Invasion_Detection轉換前/關鍵句更正.csv"

correct_df = pd.read_csv(correct_path)
fix_df = pd.read_csv(fix_path)

def convert_correct(df):
    data = []
    for _, row in df.iterrows():
        report = str(row["reportData"]).strip()
        reason = translate_text(row["理由"])
        final_code = str(row["原來答案"])  #
        output = {
            "reason": reason,
            "final_code": final_code
        }
        data.append({
            "instruction": INSTRUCTION,
            "input": report,
            "output": json.dumps(output, ensure_ascii=False)
        })
    return data

def convert_fix(df):
    data = []
    for _, row in df.iterrows():
        report = str(row["reportData"]).strip()
        reason = translate_text(row["更正後理由"])
        final_code = str(row["原來答案"])  
        output = {
            "reason": reason,
            "final_code": final_code
        }
        data.append({
            "instruction": INSTRUCTION,
            "input": report,
            "output": json.dumps(output, ensure_ascii=False)
        })
    return data

all_data = convert_correct(correct_df) + convert_fix(fix_df)

with open("train.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f" Converted {len(all_data)} samples to train.json (reason + final_code).")
