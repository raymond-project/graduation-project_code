# -*- coding: utf-8 -*-
"""
Batch 版（句子為單位）：
PPR → Graph-MMR/覆蓋 的種子選擇 + 子圖產生
→ 每個句子只輸出一筆樣本（多 seed 合併成一個子圖）
"""

from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from typing import List, Tuple, Dict
import os, re, unicodedata, json
import pandas as pd
import math, collections

# ========= 0) Neo4j 連線 =========
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "xz105923"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_all_entity_names() -> List[str]:
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN n.name AS name")
        return [r["name"] for r in result]

def build_vocab_from_neo4j() -> Tuple[Dict[str,int], Dict[str,int]]:
    """直接從 Neo4j 建 ent2id / rel2id"""
    with driver.session() as session:
        ents = [r["name"] for r in session.run("MATCH (n:Entity) RETURN DISTINCT n.name AS name")]
        rels = [r["t"]    for r in session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")]
    ent2id = {e:i for i,e in enumerate(sorted(set(ents)))}
    rel2id = {r:i for i,r in enumerate(sorted(set(rels)))}
    return ent2id, rel2id

# ========= 1) ClinicalBERT =========
MODEL_PATH = "/home/st426/system/global_graph/ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).eval().to(device)

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0, :]  # [CLS]
        vecs.append(out.cpu().numpy())
    vecs = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

# ========= 2) HNSW =========
def build_hnsw_index(embs: np.ndarray, M: int = 32, efC: int = 200, efS: int = 128):
    d = embs.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efC
    index.add(embs)
    index.hnsw.efSearch = efS
    return index

# ========= 2.5) 工具 =========
TAIL_PUNCT = " .;:?!，、。；！？"
def strip_trailing_punct(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.rstrip(TAIL_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ========= 3) PPR + Graph-MMR =========
def build_local_subgraph_from_candidates(cand_names: List[str], hops=2, max_pairs=300000):
    if not cand_names:
        return [], []
    cypher = f"""
    MATCH (s:Entity) WHERE s.name IN $cands
    MATCH p=(s)-[*1..{hops}]-(n:Entity)
    WITH collect(DISTINCT n) AS nodes
    UNWIND nodes AS a
    MATCH (a)-[r]-(b:Entity)
    WHERE b IN nodes
    RETURN DISTINCT a.name AS src, b.name AS dst
    LIMIT $limit
    """
    edges, nodeset = [], set()
    with driver.session() as session:
        res = session.run(cypher, cands=cand_names, limit=max_pairs)
        for r in res:
            u = r["src"]; v = r["dst"]
            if u != v:
                edges.append((u, v))
                nodeset.add(u); nodeset.add(v)
    return sorted(nodeset), edges

def row_normalized_transition(nodes, edges):
    adj = collections.defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    for n in nodes: _ = adj[n]
    probs_map = {n: np.full(len(adj[n]) or 1, 1.0/(len(adj[n]) or 1), dtype=np.float32) for n in nodes}
    adj = {k: sorted(list(v)) for k,v in adj.items()}
    return adj, probs_map

def softmax_temp(x: np.ndarray, tau=0.07):
    if x.size == 0: return x
    x = (x - x.max()) / max(1e-8, (x.std() + 1e-8))
    x = x / max(1e-8, tau)
    e = np.exp(x - x.max()); return e / (e.sum() + 1e-8)

def personalized_pagerank(nodes, adj, probs_map, personalization, alpha=0.2, max_iter=50, tol=1e-6):
    if not nodes: return {}
    name2idx = {n:i for i,n in enumerate(nodes)}; N = len(nodes)
    s = np.zeros(N); denom = sum(v for k,v in personalization.items() if k in name2idx)
    for k,v in personalization.items():
        if k in name2idx: s[name2idx[k]] = v
    s = s/denom if denom>0 else np.ones(N)/N
    r = s.copy()
    for _ in range(max_iter):
        r_new = np.zeros_like(r)
        for u in nodes:
            u_idx = name2idx[u]; share = (1-alpha)*r[u_idx]
            for p,v in zip(probs_map[u], adj[u]): r_new[name2idx[v]] += share*p
        r_new += alpha*s
        if np.linalg.norm(r_new-r,1)<tol: break
        r=r_new
    return {n: float(r[name2idx[n]]) for n in nodes}

def bfs_all_pairs_shortest_hops(nodes, adj, limit_hops=6):
    dist={}; node_set=set(nodes)
    for src in nodes:
        q=collections.deque([(src,0)]); seen={src}; dmap={src:0}
        while q:
            u,d=q.popleft()
            if d>=limit_hops: continue
            for v in adj.get(u,[]):
                if v not in seen:
                    seen.add(v); dmap[v]=d+1; q.append((v,d+1))
        dist[src]={v:d for v,d in dmap.items() if v in node_set}
    return dist

def graph_mmr_select_by_distance(ranked_nodes, ppr_scores, dist, k=10, lam=0.4, sigma=2.0):
    if not ranked_nodes: return []
    vals=np.array([ppr_scores.get(n,0.0) for n in ranked_nodes]); vals=vals/vals.max() if vals.max()>0 else vals
    norm={n:float(v) for n,v in zip(ranked_nodes,vals)}; selected=[]; pool=ranked_nodes.copy()
    while pool and len(selected)<k:
        best_n=None; best_score=-1e9
        for n in pool[:200]:
            red=max((math.exp(-float(d)/max(1e-6,sigma)) for s in selected if (d:=dist.get(s,{}).get(n)) is not None), default=0.0)
            score=(1-lam)*norm.get(n,0.0)-lam*red
            if score>best_score: best_score, best_n=score,n
        if not best_n: break
        selected.append(best_n); pool.remove(best_n)
    return selected

# ========= 4) 子圖 & 命中 =========
final_codes = ["000","001~979","980","987","988","990","991"]
def is_terminal_name(name: str) -> bool:
    return (name in final_codes) or (re.fullmatch(r"\d{3}", str(name)) is not None)

def paths_to_triples_from_seed(seed_name: str, max_hops=6, limit=10):
    cypher=f"""
    MATCH p=(s:Entity {{name:$seed}})-[*1..{max_hops}]->(c:Entity)
    WHERE c.name IN $codes
    RETURN p LIMIT $limit
    """
    cypher2=f"""
    MATCH p=(s:Entity {{name:$seed}})-[*1..{max_hops}]-(c:Entity)
    WHERE c.name IN $codes
    RETURN p LIMIT $limit
    """
    with driver.session() as session:
        def collect(res):
            rows=[]
            for r in res:
                path=r["p"]; nodes=[n["name"] for n in path.nodes]
                rels=[getattr(rel,"type",rel.__class__.__name__) for rel in path.relationships]
                rows.extend([(h,r,t) for h,r,t in zip(nodes[:-1],rels,nodes[1:])])
            return rows
        res=session.run(cypher,seed=seed_name,codes=final_codes,limit=limit); triples=collect(res)
        if not triples: triples=collect(session.run(cypher2,seed=seed_name,codes=final_codes,limit=limit))
    return triples

def build_seed_subgraph(seed_name,q_vec,threshold=0.65):
    raw=paths_to_triples_from_seed(seed_name)
    if not raw: return []
    terminal_nodes={t for _,_,t in raw if is_terminal_name(t)}
    all_nodes=[n for h,_,t in raw for n in (h,t)]
    uniq=sorted(set(all_nodes))
    vecs=embed_texts(uniq); sims=vecs@q_vec
    keep={n:float(sim) for n,sim in zip(uniq,sims) if sim>=threshold}
    keep[seed_name]=1.0
    for t in terminal_nodes: keep[t]=1.0
    return [(h,r,t) for h,r,t in raw if (is_terminal_name(t) or (h in keep and t in keep))]

def normalize_truth_code(val)->str:
    if pd.isna(val): return ""
    s=str(val).strip()
    if re.fullmatch(r"\d+",s): num=int(s); return f"{num:03d}" if 0<=num<=999 else s
    return s

SET_CODES={"000","980","987","988","990","991"}
def is_match_by_name(reached: set, truth: str)->bool:
    if not truth: return False
    if truth in SET_CODES: return truth in reached
    if re.fullmatch(r"\d{3}",truth):
        v=int(truth)
        if 1<=v<=979: return "001~979" in reached
        if v==0: return "000" in reached
        return truth in reached
    return truth in reached

# ========= 5) 主程式 =========
if __name__=="__main__":
    ent2id, rel2id = build_vocab_from_neo4j()
    id2ent={v:k for k,v in ent2id.items()}

    node_names=get_all_entity_names()
    print(f"取得 {len(node_names)} 個節點名稱，開始產生節點向量…")
    node_vecs=embed_texts(node_names,batch_size=64)
    index=build_hnsw_index(node_vecs); name2idx={n:i for i,n in enumerate(node_names)}

    CSV_IN="/home/st426/system/global_graph/預測結果_正確_surdis.csv"
    CSV_OUT="/home/st426/system/global_graph/預測結果_無匹配.csv"
    JSONL_OUT="/home/st426/system/global_graph/gnn_train.jsonl"

    df=pd.read_csv(CSV_IN)
    sentences=[strip_trailing_punct(s) for s in df["sentence"].fillna("").astype(str)]
    q_vecs=embed_texts(sentences,batch_size=128)

    topK=200; SEED_K=5
    out_rows=[]; json_count=0; matched_count=0

    with open(JSONL_OUT,"w",encoding="utf-8") as jsonl_f:
        for i,(qv_row,row) in enumerate(zip(q_vecs,df.itertuples(index=False))):
            truth_code=normalize_truth_code(getattr(row,"原發部位手術切緣距離"))
            sentence=getattr(row,"sentence"); report=getattr(row,"reportData")

            D,I=index.search(qv_row.reshape(1,-1),min(topK,node_vecs.shape[0]))
            cand_ids=[idx for idx in I[0].tolist() if idx!=-1]
            cand_names=[node_names[s] for s in cand_ids]

            nodes_sub,edges_sub=build_local_subgraph_from_candidates(cand_names)
            if not nodes_sub: nodes_sub=cand_names.copy(); edges_sub=[]
            adj,probs_map=row_normalized_transition(nodes_sub,edges_sub)

            sims=[float(node_vecs[name2idx[n]]@qv_row) for n in cand_names if n in adj]
            cand_in_sub=[n for n in cand_names if n in adj]
            personalization={n:w for n,w in zip(cand_in_sub,softmax_temp(np.array(sims),tau=0.07))} if cand_in_sub else {n:1.0 for n in nodes_sub}

            ppr=personalized_pagerank(nodes_sub,adj,probs_map,personalization)
            ranked=sorted(nodes_sub,key=lambda n:ppr.get(n,0.0),reverse=True)
            dist=bfs_all_pairs_shortest_hops(ranked[:200],adj)
            seed_nodes=graph_mmr_select_by_distance(ranked,ppr,dist,k=SEED_K)

            kept_triples=[]
            for seed in seed_nodes: kept_triples.extend(build_seed_subgraph(seed,q_vec=qv_row))

            reached={t for _,_,t in kept_triples if is_terminal_name(t)}
            matched=is_match_by_name(reached,truth_code)

            if not matched:
                out_rows.append({"原發部位手術切緣距離":getattr(row,"原發部位手術切緣距離"),"sentence":sentence,"reportData":report})
            else:
                agg_tuples=[]; agg_nodes=set()
                for h,r,t in kept_triples:
                    h_id,r_id,t_id=ent2id.get(h),rel2id.get(r),ent2id.get(t)
                    if None in (h_id,r_id,t_id): continue
                    agg_tuples.append([h_id,r_id,t_id]); agg_nodes.update([h_id,t_id])

                start_ids=[ent2id[n] for n in seed_nodes if n in ent2id]
                if not start_ids and cand_ids: 
                    fb_id=ent2id.get(node_names[cand_ids[0]])
                    if fb_id is not None: start_ids=[fb_id]

                answer_id=None
                if truth_code:
                    if truth_code in ent2id: answer_id=ent2id[truth_code]
                    elif re.fullmatch(r"\d{3}",truth_code):
                        v=int(truth_code)
                        if 1<=v<=979 and "001~979" in ent2id: answer_id=ent2id["001~979"]
                if answer_id is None and truth_code in {"000","980","987","988","990","991"}: answer_id=ent2id.get(truth_code)

                sample={"id":f"sur_dis-{json_count}","question":sentence,"entities":start_ids,
                        "answers":[{"kb_id":str(answer_id),"text":truth_code}] if answer_id else [],
                        "subgraph":{"tuples":agg_tuples,"entities":sorted(list(agg_nodes))}}
                jsonl_f.write(json.dumps(sample,ensure_ascii=False)+"\n")
                json_count+=1; matched_count+=1

            if i%50==0: print(f"[CSV] 已處理 {i} 筆；不匹配 {len(out_rows)}；匹配 {matched_count}")

    pd.DataFrame(out_rows).to_csv(CSV_OUT,index=False)
    print(f"[CSV] 完成。不匹配 {len(out_rows)} → {CSV_OUT}")
    print(f"[JSONL] 訓練樣本（匹配） {matched_count} → {JSONL_OUT}")
