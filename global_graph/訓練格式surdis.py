# -*- coding: utf-8 -*-
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from typing import List, Tuple, Dict
import os, re, unicodedata, json
import pandas as pd

# ========= 0) Neo4j =========
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "xz105923"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_all_entity_names() -> List[str]:
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN n.name AS name")
        return [r["name"] for r in result]

# ========= 0.5) 檔案路徑 =========
JSON_PATH = "/home/st426/system/global_graph/surgical_margin_graph.json"
ENTITIES_TXT = "/home/st426/system/GNN-RAG/gnn/data/surdis3/entities.txt"
RELATIONS_TXT = "/home/st426/system/GNN-RAG/gnn/data/surdis3/relations.txt"
VOCAB_TXT = "/home/st426/system/GNN-RAG/gnn/data/surdis3/vocab.txt"

# ========= 0.6) 由 JSON 建立 vocab =========
def build_vocab_from_graph(json_path, entities_txt, relations_txt, vocab_txt):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data if isinstance(data, list) else data.get("triples", [])
    entities, relations = set(), set()
    for t in triples:
        subj, rel, obj = t.get("subject",""), t.get("relation",""), t.get("object","")
        if subj: entities.add(subj)
        if obj:  entities.add(obj)
        if rel:  relations.add(rel)
    if not os.path.exists(entities_txt):
        with open(entities_txt,"w",encoding="utf-8") as f:
            for e in sorted(entities): f.write(e+"\n")
    if not os.path.exists(relations_txt):
        with open(relations_txt,"w",encoding="utf-8") as f:
            for r in sorted(relations): f.write(r+"\n")
    if not os.path.exists(vocab_txt):
        vocab=set()
        for e in entities:
            for tok in str(e).split():
                vocab.add(tok)
        with open(vocab_txt,"w",encoding="utf-8") as f:
            for v in sorted(vocab): f.write(v+"\n")

# ========= 1) ClinicalBERT =========
MODEL_PATH = "/home/st426/system/global_graph/ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).eval().to(device)

def embed_texts(texts: List[str], batch_size=64) -> np.ndarray:
    vecs=[]
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        inputs=tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            out=model(**inputs).last_hidden_state[:,0,:]
        vecs.append(out.cpu().numpy())
    vecs=np.vstack(vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

# ========= 2) HNSW =========
def build_hnsw_index(embs: np.ndarray, M=32, efC=200, efS=128):
    d=embs.shape[1]
    index=faiss.IndexHNSWFlat(d,M)
    index.hnsw.efConstruction=efC
    index.add(embs)
    index.hnsw.efSearch=efS
    return index

# ========= 2.5) 去標點 =========
TAIL_PUNCT = " .;:?!，、。；！？"
def strip_trailing_punct(s: str) -> str:
    if not isinstance(s,str): s="" if s is None else str(s)
    s=unicodedata.normalize("NFKC",s)
    s=s.rstrip(TAIL_PUNCT)
    return re.sub(r"\s+"," ",s).strip()

# ========= 3) Neo4j 子圖 =========
final_codes=["000","001~979","980","987","988","990","991"]

def is_terminal_name(name: str) -> bool:
    return (name in final_codes) or (re.fullmatch(r"\d{3}",str(name)) is not None)

def paths_to_triples_from_seed(seed_name, max_hops=6, limit=10):
    cypher=f"""
    MATCH p = (s:Entity {{name:$seed}})-[*1..{max_hops}]->(c:Entity)
    WHERE c.name IN $codes
    RETURN p LIMIT {limit}
    """
    triples=[]
    with driver.session() as session:
        res=session.run(cypher, seed=seed_name, codes=final_codes)
        for r in res:
            path=r["p"]
            nodes=[n["name"] for n in path.nodes]
            rels=[rel.type for rel in path.relationships]
            triples.extend([(h,r,t) for h,r,t in zip(nodes[:-1], rels, nodes[1:])])
    return triples

def build_seed_subgraph(seed_name,q_vec,threshold=0.65):
    raw=paths_to_triples_from_seed(seed_name, max_hops=6, limit=10)
    if not raw:
        return {"seed":seed_name,"seed_id":ent2id.get(seed_name),"tuples":[],"entities":[]}
    kept_triples=[]
    for h,r,t in raw:
        h_id, r_id, t_id = ent2id.get(h), rel2id.get(r), ent2id.get(t)
        if None in (h_id,r_id,t_id): continue
        kept_triples.append([h_id,r_id,t_id])
    ent_ids=set()
    for h_id,_,t_id in kept_triples:
        ent_ids.update([h_id,t_id])
    return {"seed":seed_name,"seed_id":ent2id.get(seed_name),
            "tuples":kept_triples,"entities":sorted(ent_ids)}

# ========= 4) Vocab =========
def load_vocab(path):
    mapping={}
    with open(path,"r",encoding="utf-8") as f:
        for idx,line in enumerate(f):
            name=line.strip()
            if name: mapping[name]=idx
    return mapping

# ========= 5) Label =========
def normalize_truth_code(val):
    if pd.isna(val): return ""
    s=str(val).strip()
    if re.fullmatch(r"\d+",s):
        num=int(s)
        if 0<=num<=999: return f"{num:03d}"
        return s
    return s

SET_CODES={"000","980","987","988","990","991"}
def is_match(reached_codes, truth_code: str) -> bool:
    if not truth_code: return False
    reached=set(reached_codes)
    if truth_code in SET_CODES: return truth_code in reached
    if re.fullmatch(r"\d{3}",truth_code):
        v=int(truth_code)
        if 1<=v<=979: return "001~979" in reached
        if v==0: return "000" in reached
        return truth_code in reached
    return truth_code in reached

# ========= MAIN =========
if __name__=="__main__":
    if (not os.path.exists(ENTITIES_TXT)) or (not os.path.exists(RELATIONS_TXT)) or (not os.path.exists(VOCAB_TXT)):
        build_vocab_from_graph(JSON_PATH,ENTITIES_TXT,RELATIONS_TXT,VOCAB_TXT)
    ent2id=load_vocab(ENTITIES_TXT)
    rel2id=load_vocab(RELATIONS_TXT)

    node_names=get_all_entity_names()
    print(f"取得 {len(node_names)} 個節點名稱，開始產生向量…")
    node_vecs=embed_texts(node_names,batch_size=64)
    index=build_hnsw_index(node_vecs)

    CSV_IN="/home/st426/system/global_graph/預測結果_正確_surdis.csv"
    CSV_OUT="/home/st426/system/global_graph/預測結果_無匹配.csv"
    JSONL_OUT="/home/st426/system/GNN-RAG/gnn/data/surdis3/gnn_train.jsonl"
    

    df=pd.read_csv(CSV_IN)
    sentences_raw=df["sentence"].fillna("").astype(str).tolist()
    sentences_norm=[strip_trailing_punct(s) for s in sentences_raw]
    q_vecs=embed_texts(sentences_norm,batch_size=128)

    out_rows=[]; json_count=0
    with open(JSONL_OUT,"w",encoding="utf-8") as jsonl_f:
        for i,(qv_row,row) in enumerate(zip(q_vecs, df.itertuples(index=False))):
            truth_code=normalize_truth_code(getattr(row,"原發部位手術切緣距離"))
            sentence=getattr(row,"sentence")
            report=getattr(row,"reportData")

            D,I=index.search(qv_row.reshape(1,-1),50)
            cand_ids=[idx for idx in I[0].tolist() if idx!=-1]
            seed_nodes_batch=[node_names[i] for i in cand_ids[:5]]

            matched_any=False; samples_this_sentence=[]
            for j,seed_name in enumerate(seed_nodes_batch):
                sg=build_seed_subgraph(seed_name,q_vec=qv_row,threshold=0.65)
                id2ent={v:k for k,v in ent2id.items()}
                reached_codes={id2ent.get(t_id,"") for _,_,t_id in sg["tuples"]
                               if is_terminal_name(id2ent.get(t_id,""))}
                if is_match(reached_codes,truth_code):
                    matched_any=True
                    answer_id = None
                    if truth_code:
                        if truth_code in ent2id:
                            answer_id = ent2id[truth_code]
                        elif re.fullmatch(r"\d{3}", truth_code):  # e.g. "040"
                            v = int(truth_code)
                            if 1 <= v <= 979 and "001~979" in ent2id:
                                answer_id = ent2id["001~979"]
                            elif v == 0 and "000" in ent2id:
                                answer_id = ent2id["000"]

                    sample = {
                        "id": f"sur_dis-{json_count}-{j}",
                        "question": sentence,
                        "entities": [ent2id[n] for n in seed_nodes_batch if n in ent2id],  # ← 全部 5 個
                        "answers": [{"kb_id": str(answer_id), "text": truth_code}] if answer_id is not None else [],
                        "subgraph": {"tuples": sg["tuples"], "entities": sg["entities"]}
                    }


                    samples_this_sentence.append(sample)

            if matched_any:
                for s in samples_this_sentence:
                    jsonl_f.write(json.dumps(s,ensure_ascii=False)+"\n")
                json_count+=1
            else:
                out_rows.append({"原發部位手術切緣距離":getattr(row,"原發部位手術切緣距離"),
                                 "sentence":sentence,"reportData":report})

            if i%50==0:
                print(f"[CSV] 已處理 {i+1} 筆；不匹配 {len(out_rows)}；匹配 {json_count}")

    pd.DataFrame(out_rows).to_csv(CSV_OUT,index=False)
    print(f"[CSV] 不匹配：{len(out_rows)} → {CSV_OUT}")
    print(f"[JSONL] 訓練樣本：{json_count} → {JSONL_OUT}")

    driver.close()
