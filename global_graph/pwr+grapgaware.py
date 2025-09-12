# -*- coding: utf-8 -*-
"""
PPR → Graph-MMR/覆蓋 的種子選擇流程
- 先用 ClinicalBERT+HNSW 取語意候選
- 在候選的 k-hop 子圖上做 Personalized PageRank（RWR）
- 之後用 Graph-MMR（依圖距離做去冗/覆蓋）挑出 seeds
- 再用 seeds 走路徑到終端碼、與取小鄰域做檢查
"""

from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple
import re, unicodedata, math, collections

# ========= 0) Neo4j 連線 =========
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "xz105923"
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_all_entity_names() -> List[str]:
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN n.name AS name")
        return [r["name"] for r in result]

# ========= 1) ClinicalBERT =========
MODEL_PATH = "/home/st426/system/global_graph/ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).eval().to(device)

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """CLS sentence embedding（L2 normalize → 可用內積等效 cosine）"""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0, :]  # [CLS]
        vecs.append(out.cpu().numpy())
    vecs = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

# ========= 2) HNSW =========
def build_hnsw_index(embs: np.ndarray, M: int = 32, efC: int = 200, efS: int = 128) -> faiss.IndexHNSWFlat:
    d = embs.shape[1]
    index = faiss.IndexHNSWFlat(d, M)   # L2 metric（向量已 L2 normalize → 等效 cosine）
    index.hnsw.efConstruction = efC
    index.add(embs)
    index.hnsw.efSearch = efS
    return index

# ========= 2.5) 前處理 =========
TAIL_PUNCT = " .;:?!，、。；！？"
def strip_trailing_punct(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.rstrip(TAIL_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ========= 3) PPR（RWR） =========
def build_local_subgraph_from_candidates(cand_names: List[str], hops: int = 2, max_pairs: int = 300000) -> Tuple[List[str], List[Tuple[str,str]]]:
    """
    從候選節點擴展 k-hop 的局部子圖（無向邊），回傳 (nodes, edges)
    注意：這裡採無向，利於距離與擴散（走路徑仍用有向查）
    """
    if not cand_names:
        return [], []

    cypher = f"""
    MATCH (s:Entity)
    WHERE s.name IN $cands
    MATCH p=(s)-[*1..{hops}]-(n:Entity)
    WITH collect(DISTINCT n) AS nodes
    UNWIND nodes AS a
    MATCH (a)-[r]-(b:Entity)
    WHERE b IN nodes
    RETURN DISTINCT a.name AS src, b.name AS dst
    LIMIT $limit
    """
    edges = []
    nodeset = set()
    with driver.session() as session:
        res = session.run(cypher, cands=cand_names, limit=max_pairs)
        for r in res:
            u = r["src"]; v = r["dst"]
            if u == v: 
                continue
            edges.append((u, v))
            nodeset.add(u); nodeset.add(v)
    return sorted(nodeset), edges

def row_normalized_transition(nodes: List[str], edges: List[Tuple[str,str]]) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """
    以無向圖建轉移機率；每個節點均勻分給鄰居；無出邊者加 self-loop
    回傳鄰接清單 adj 以及每個節點的鄰居機率向量 probs_map（與 adj 對齊）
    """
    adj = collections.defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    for n in nodes:
        _ = adj[n]

    probs_map: Dict[str, np.ndarray] = {}
    for n in nodes:
        nbrs = sorted(adj[n])
        if len(nbrs) == 0:
            adj[n].add(n)
            nbrs = [n]
        probs = np.full(len(nbrs), 1.0 / len(nbrs), dtype=np.float32)
        probs_map[n] = probs
    adj = {k: sorted(list(v)) for k, v in adj.items()}
    return adj, probs_map

def softmax(x: np.ndarray, tau: float = 0.07) -> np.ndarray:
    x = (x - x.max()) / max(1e-8, (x.std() + 1e-8))
    x = x / max(1e-8, tau)
    m = x.max()
    e = np.exp(x - m)
    s = e.sum()
    return e / (s + 1e-8)

def personalized_pagerank(
    nodes: List[str],
    adj: Dict[str, List[str]],
    probs_map: Dict[str, np.ndarray],
    personalization: Dict[str, float],
    alpha: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-6
) -> Dict[str, float]:
    if not nodes:
        return {}
    name2idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)

    s = np.zeros(N, dtype=np.float64)
    if len(personalization) > 0:
        vals = np.array([v for k,v in personalization.items() if k in name2idx], dtype=np.float64)
        denom = vals.sum()
        for k,v in personalization.items():
            if k in name2idx:
                s[name2idx[k]] = v
        if denom > 0:
            s /= denom
        else:
            s[:] = 1.0 / N
    else:
        s[:] = 1.0 / N

    r = s.copy()
    for _ in range(max_iter):
        r_new = np.zeros_like(r)
        for u in nodes:
            u_idx = name2idx[u]
            nbrs = adj[u]
            probs = probs_map[u]
            if len(nbrs) == 0:
                r_new[u_idx] += (1 - alpha) * r[u_idx]
            else:
                share = (1 - alpha) * r[u_idx]
                for p, v in zip(probs, nbrs):
                    r_new[name2idx[v]] += share * p
        r_new += alpha * s
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    return {n: float(r[name2idx[n]]) for n in nodes}

# ========= 4) Graph-MMR/覆蓋（用圖距離做去冗） =========
def bfs_all_pairs_shortest_hops(nodes: List[str], adj: Dict[str, List[str]], limit_hops: int = 6) -> Dict[str, Dict[str, int]]:
    dist = {}
    node_set = set(nodes)
    for src in nodes:
        q = collections.deque([(src, 0)])
        seen = {src}
        dmap = {src: 0}
        while q:
            u, d = q.popleft()
            if d >= limit_hops:
                continue
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dmap[v] = d + 1
                    q.append((v, d + 1))
        dist[src] = {v:d for v,d in dmap.items() if v in node_set}
    return dist

def graph_mmr_select_by_distance(
    ranked_nodes: List[str],
    ppr_scores: Dict[str, float],
    dist: Dict[str, Dict[str, int]],
    k: int = 10,
    lam: float = 0.4,
    sigma: float = 2.0
) -> List[str]:
    if not ranked_nodes:
        return []
    vals = np.array([ppr_scores.get(n, 0.0) for n in ranked_nodes], dtype=np.float64)
    if vals.max() > 0:
        vals = vals / vals.max()
    norm = {n: float(v) for n, v in zip(ranked_nodes, vals)}

    selected = []
    pool = ranked_nodes.copy()

    while pool and len(selected) < k:
        best_n = None
        best_score = -1e9
        for n in pool[:200]:
            if not selected:
                red = 0.0
            else:
                cand = []
                for s in selected:
                    d = dist.get(s, {}).get(n, None)
                    if d is None:
                        continue
                    cand.append(math.exp(- float(d) / max(1e-6, sigma)))
                red = max(cand) if cand else 0.0
            score = (1.0 - lam) * norm.get(n, 0.0) - lam * red
            if score > best_score:
                best_score = score
                best_n = n
        if best_n is None:
            break
        selected.append(best_n)
        pool.remove(best_n)
    return selected

# ========= 5) 與終端碼路徑 / 鄰域檢視 =========
final_codes = ["000","001~979","980","987","988","990","991"]

def paths_to_final_codes_from_seed(seed_name: str, q_vec: np.ndarray, max_hops: int = 6, limit: int = 10):
    cypher = f"""
    MATCH p = (s:Entity {{name:$seed}})-[*1..{max_hops}]->(c:Entity)
    WHERE c.name IN $codes
    RETURN c.name AS final_code, p
    ORDER BY length(p) ASC
    LIMIT $limit
    """
    with driver.session() as session:
        res = session.run(
            cypher,
            seed=seed_name,
            codes=final_codes,
            limit=limit,
        )
        results = []
        for r in res:
            path = r["p"]
            node_names_in_path = [n["name"] for n in path.nodes]
            rel_types = [type(rel).__name__ for rel in path.relationships]
            results.append({
                "final_code": r["final_code"],
                "nodes": node_names_in_path,
                "rels": rel_types
            })
        return results

def small_neighborhood(seed_name: str, hops: int = 2, limit_paths: int = 30):
    cypher = f"""
    MATCH p = (s:Entity {{name:$seed}})-[*1..{hops}]-(n:Entity)
    RETURN p
    LIMIT {limit_paths}
    """
    with driver.session() as session:
        res = session.run(cypher, seed=seed_name)
        all_nodes = []
        for r in res:
            all_nodes.extend([n["name"] for n in r["p"].nodes])
        all_nodes = sorted(list(set(all_nodes)))
    return {"all_nodes": all_nodes}

def select_best_path(paths: List[Dict], q_vec: np.ndarray) -> Dict:
    """挑出最符合 query 的路徑（純語意相似度，不加長度懲罰）"""
    best_p, best_score = None, -1e9
    for p in paths:
        node_texts = p["nodes"]
        node_vecs = embed_texts(node_texts)
        avg_vec = node_vecs.mean(axis=0)
        score = float(avg_vec @ q_vec)   
        if score > best_score:
            best_score, best_p = score, p
    if best_p:
        best_p["score"] = best_score
    return best_p

# ========= 主流程 =========
if __name__ == "__main__":
    node_names = get_all_entity_names()
    if not node_names:
        raise RuntimeError("Neo4j 沒抓到任何 (Entity) 節點名稱，請先確認資料已匯入。")

    print(f"取得 {len(node_names)} 個節點名稱，開始產生節點向量…")
    node_vecs = embed_texts(node_names, batch_size=64)

    print(" 建立 HNSW 索引（近似；L2，向量已單位化 → 等效 cosine）…")
    index = build_hnsw_index(node_vecs, M=32, efC=200, efS=128)

    # ===== 查詢 =====
       # ===== 查詢 =====
          # ===== 查詢 =====
             # ===== 查詢 =====
                # ===== 查詢 =====
                   # ===== 查詢 =====
                      # ===== 查詢 =====
                         # ===== 查詢 =====
                            # ===== 查詢 =====
                            
    query_text = "The resection margins are free of lesion."
    q_norm = strip_trailing_punct(query_text)
    print("\nQuery:", q_norm)
    qv = embed_texts([q_norm])[0]

    topK = min(200, node_vecs.shape[0])
    D, I = index.search(qv.reshape(1, -1), topK)
    cand_ids = [idx for idx in I[0].tolist() if idx != -1]
    if not cand_ids:
        raise RuntimeError("沒有語意候選節點（檢查嵌入或索引）。")

    cand_names = [node_names[i] for i in cand_ids]
    cand_vecs = node_vecs[cand_ids]
    sims = cand_vecs @ qv

    HOPS_FOR_SUBGRAPH = 2
    nodes_sub, edges_sub = build_local_subgraph_from_candidates(cand_names, hops=HOPS_FOR_SUBGRAPH)
    print(f"局部子圖：節點 {len(nodes_sub)} / 邊 {len(edges_sub)}")

    if not nodes_sub:
        nodes_sub = cand_names.copy()
        edges_sub = []

    adj, probs_map = row_normalized_transition(nodes_sub, edges_sub)

    cand_in_sub = [n for n in cand_names if n in adj]
    sims_for_sub = np.array([float(sims[cand_names.index(n)]) for n in cand_in_sub], dtype=np.float32) if cand_in_sub else np.zeros(0)
    if len(cand_in_sub) == 0:
        personalization = {n: 1.0 for n in nodes_sub}
    else:
        probs = softmax(sims_for_sub, tau=0.07)
        personalization = {n: float(p) for n, p in zip(cand_in_sub, probs)}

    ppr = personalized_pagerank(
        nodes_sub, adj, probs_map, personalization,
        alpha=0.2, max_iter=50, tol=1e-6
    )
    ranked = sorted(nodes_sub, key=lambda n: ppr.get(n, 0.0), reverse=True)

    print("\nTop-20 PPR 節點：")
    for i, n in enumerate(ranked[:20], 1):
        print(f"{i:>2}. {n}  ppr={ppr[n]:.6f}")

    TOP_FOR_DISTANCE = 200
    for_dist = ranked[:TOP_FOR_DISTANCE]
    dist = bfs_all_pairs_shortest_hops(for_dist, adj, limit_hops=6)

    SEED_K =10
    seeds = graph_mmr_select_by_distance(
        ranked_nodes=ranked,
        ppr_scores=ppr,
        dist=dist,
        k=SEED_K,
        lam=0.4,
        sigma=2.0
    )
    print(f"\n 最終種子（Graph-MMR over PPR，k={SEED_K}）：")
    for r, s in enumerate(seeds, 1):
        print(f"{r:>2}. {s}  ppr={ppr.get(s,0.0):.6f}")

    # 每個 seed 自己找最佳路徑 + 鄰域
    for seed in seeds:
        print(f"\n==== Seed: {seed} ====")
        paths = paths_to_final_codes_from_seed(seed, qv, max_hops=6, limit=10)
        if not paths:
            print("  沒有找到通往終端編碼的路徑")
        else:
            best_path = select_best_path(paths, qv)
            if best_path:
                print(f"  最佳路徑 (score={best_path['score']:.4f}):")
                print(f"    FinalCode: {best_path['final_code']}")
                print("    Nodes:", " -> ".join(best_path["nodes"]))
                print("    Rels :", " -> ".join(best_path["rels"]))

        neigh = small_neighborhood(seed, hops=2, limit_paths=30)
        print(f"  子圖鄰域節點數（2-hop）: {len(neigh['all_nodes'])}")
