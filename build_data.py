# -*- coding: utf-8 -*-
"""
构建预计算数据 — 本地运行一次，生成 data/documents.json + data/embeddings.npz
之后将 data/ 目录一起提交到仓库，Vercel 冷启动时直接加载
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import RAGEngine, Document
from rag_engine import EmbeddingIndex


def build():
    source_files = [
        "深圳市社保法律法规汇编.txt",
        "深圳市住房公积金法律法规汇编.txt",
    ]

    # 1. 解析文档
    engine = RAGEngine()
    docs = engine.build_from_txt_files(source_files, encoding='gb18030')

    # 2. 序列化文档元数据
    doc_data = []
    for doc in docs:
        doc_data.append({
            "id": doc.id,
            "content": doc.content,
            "chunk_type": doc.chunk_type,
            "parent_id": doc.parent_id,
            "category": doc.category,
            "metadata": doc.metadata,
        })

    os.makedirs("data", exist_ok=True)
    with open("data/documents.json", "w", encoding="utf-8") as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(doc_data)} documents to data/documents.json")

    # 3. 编码 child 文档并保存向量
    child_docs = [d for d in docs if d.chunk_type == "child"]
    if child_docs:
        emb_idx = EmbeddingIndex()
        texts = [d.content for d in child_docs]
        embeddings = emb_idx.encode_documents_batch(texts)

        child_ids = [str(i) for i in range(len(child_docs))]
        emb_matrix = np.array(embeddings, dtype=np.float32)
        np.savez_compressed("data/embeddings.npz", ids=child_ids, embeddings=emb_matrix)
        print(f"[OK] Saved {len(embeddings)} embeddings to data/embeddings.npz")

    # 4. 复制法律文本到 data 目录（Vercel 部署时需要）
    import shutil
    for f in source_files:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join("data", os.path.basename(f)))
            print(f"[OK] Copied {f} to data/")

    print("\n[DONE] Build complete. Commit the 'data/' directory to your repo.")


if __name__ == "__main__":
    build()
