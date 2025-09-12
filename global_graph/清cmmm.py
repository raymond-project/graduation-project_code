import json
import re

input_file = "/home/st426/system/global_graph/surgical_margin_graph.json"
output_file = "/home/st426/system/global_graph/surgical_margin_graph.json"

# 1. 帶有比較符號的 (可有可無數字，都要保留)
pattern_keep = re.compile(r"[<>]=?\s*\d*(\.\d+)?\s*(mm|cm)\b")

# 2. 一般數字 + 單位 (要清掉數字)
pattern_remove = re.compile(r"\b\d+(\.\d+)?\s*(mm|cm)\b")
pattern_multiplicative = re.compile(r"\b\d+(\.\d+)?\s*[x×]\s*\d*(\.\d+)?\s*(mm|cm)\b")
pattern_range = re.compile(r"\b\d+(\.\d+)?\s*[-–]\s*(mm|cm)\b")

def clean_text(text):
    if isinstance(text, str):
        # 1. 保留比較符號的情況
        text = pattern_keep.sub(lambda m: m.group(0), text)

        # 2. 先處理 "數字 x 數字 + 單位"
        text = pattern_multiplicative.sub(r" \3", text)

        # 3. 移除單一數字 + 單位
        text = pattern_remove.sub(r" \2", text)
        text = pattern_range.sub(r" \2", text)

        return text
    return text

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def traverse(obj, parent_key=None):
    if isinstance(obj, dict):
        return {k: traverse(v, k) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [traverse(i, parent_key) for i in obj]
    elif isinstance(obj, str):
        # 特別處理 object 欄位
        if parent_key == "object":
            if obj.isdigit():
                num = int(obj)
                if 1 <= num <= 979:
                    return "001~979"
        return clean_text(obj)
    else:
        return obj

cleaned_data = traverse(data)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"清理完成，結果存到 {output_file}")
