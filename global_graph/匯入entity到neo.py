from neo4j import GraphDatabase
import json

# 連線設定
uri = "bolt://localhost:17687"
user = "neo4j"
password = "xz105923"
driver = GraphDatabase.driver(uri, auth=(user, password))
def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print(" Neo4j 資料庫已清空")
def import_triples_to_neo4j(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    with driver.session() as session:
        for t in triples:
            subject = t["subject"]
            relation = t["relation"]
            object_ = t["object"]

            # 建立 Cypher 查詢，把 relation 包在反引號裡避免語法錯誤
            cypher = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:`{relation}`]->(o)
            """

            session.run(cypher, subject=subject, object=object_)

    print(f" 已匯入 {len(triples)} 條三元組")

if __name__ == "__main__":
    clear_database()
    import_triples_to_neo4j("/home/st426/system/global_graph/surgical_margin_graph.json")
