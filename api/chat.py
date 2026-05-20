# -*- coding: utf-8 -*-
"""
Vercel serverless function — 五险一金智能助手 API
冷启动时加载 RAG 引擎（BM25 + 预计算 embedding），warm start 复用
"""

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rag_engine import RAGEngine, Document
from rag import EnhancedRAG
from memory import UserMemory

# ── 模块级全局：冷启动初始化一次，warm start 复用 ──────────
_rag = None


def _init_rag():
    global _rag
    if _rag is not None:
        return _rag

    print("[INFO] Cold start: initializing RAG engine...")

    _rag = EnhancedRAG()
    engine = _rag.rag_engine

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    # 尝试从预计算数据加载
    docs_path = os.path.join(data_dir, "documents.json")
    emb_path = os.path.join(data_dir, "embeddings.npz")

    if os.path.exists(docs_path) and os.path.exists(emb_path):
        print("[INFO] Loading pre-computed data...")

        # 加载文档
        with open(docs_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        docs = []
        for d in doc_data:
            docs.append(Document(
                id=d["id"], content=d["content"],
                chunk_type=d["chunk_type"], parent_id=d.get("parent_id", ""),
                category=d.get("category", ""), metadata=d.get("metadata"),
            ))
        engine.add_documents(docs)
        print(f"[OK] Loaded {len(docs)} documents")

        # 加载预计算向量 → numpy 数组
        data = np.load(emb_path, allow_pickle=False)
        ids = data["ids"].tolist()
        embeddings = data["embeddings"].tolist()
        engine.search_engine.embedding_index.populate_from_precomputed(ids, embeddings)
    else:
        # 回退：从法律文本文件构建 BM25 索引（embedding 需 API key）
        print("[INFO] No pre-computed data found, building BM25 index from source...")
        source_files = [
            os.path.join(data_dir, "深圳市社保法律法规汇编.txt"),
            os.path.join(data_dir, "深圳市住房公积金法律法规汇编.txt"),
        ]
        if not os.path.exists(source_files[0]):
            source_files = [
                "深圳市社保法律法规汇编.txt",
                "深圳市住房公积金法律法规汇编.txt",
            ]
        engine.rebuild_index(source_files, encoding='gb18030')

        # 如果 ZHIPU_API_KEY 存在，编码 embedding
        from rag_engine import ZHIPU_API_KEY
        if ZHIPU_API_KEY:
            child_docs = [d for d in engine.search_engine.child_docs]
            if child_docs:
                print("[INFO] Encoding embeddings via API (fallback)...")
                emb_idx = engine.search_engine.embedding_index
                texts = [d.content for d in child_docs]
                ids = [str(i) for i in range(len(child_docs))]
                embeddings = emb_idx.encode_documents_batch(texts)
                emb_idx.populate_from_precomputed(ids, embeddings)
        else:
            print("[WARN] ZHIPU_API_KEY not set, semantic search disabled (BM25 only)")

    print(f"[OK] RAG engine ready ({engine.get_stats()})")
    return _rag


# ── SSE 流式输出辅助 ──────────────────────────────────────────

def _stream_response(wfile, query: str, profile: dict, history: list):
    """调用 RAG 流式生成，将 token 写入 SSE 事件"""

    def send_event(data_str):
        wfile.write(f"data: {data_str}\n\n".encode('utf-8'))

    def cb(token):
        send_event(json.dumps({"token": token}, ensure_ascii=False))

    try:
        rag = _init_rag()
        answer, _ = rag.chat_streaming(query, cb, history, profile=profile)
        send_event(json.dumps({"done": True, "full_answer": answer}, ensure_ascii=False))
    except Exception as e:
        send_event(json.dumps({"error": str(e)}, ensure_ascii=False))


# ── HTTP Handler ──────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self._cors_headers()
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            message = body.get("message", "").strip()
            if not message:
                self._json_error(400, "message is required")
                return

            profile = body.get("profile", {})
            history = body.get("history", [])

            # SSE 流式响应
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            _stream_response(self.wfile, message, profile, history)

        except Exception:
            self._json_error(500, traceback.format_exc())

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_error(self, status, detail):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": detail}, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        # 静默日志，避免污染 stderr
        pass
