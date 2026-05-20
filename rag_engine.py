# -*- coding: utf-8 -*-
"""
RAG 引擎 — 混合检索（Vercel 版 — BM25 + 智谱 Embedding API）
─────────────────────────────
稀疏信号: BM25 (jieba 分词)     — 关键词精准匹配
稠密信号: Dense Embedding       — 语义理解
融合方式: RRF (Reciprocal Rank Fusion)

检索顺序: 先子后父（子文档精确检索 → 父文档获取上下文）
上下文窗口: 匹配子文档 ± 邻居子文档
"""

import os
import re
import math
import time
import json
import urllib.request
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ── 分词 ──────────────────────────────────────────────────────
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

# ── 稠密检索（智谱 Embedding API）─────────────────────────────
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_EMBEDDING_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/embeddings"
ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")

HAS_EMBEDDING = bool(ZHIPU_API_KEY)
EMBEDDING_MODEL_NAME = ZHIPU_EMBEDDING_MODEL


@dataclass
class Document:
    id: str
    content: str
    chunk_type: str
    parent_id: str = ""
    category: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ═══════════════════════════════════════════════════════════════
# 分词工具
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    if HAS_JIEBA:
        words = jieba.lcut(text)
        return [w.strip() for w in words if len(w.strip()) >= 2]
    words = []
    for seq in re.findall(r'[一-龥]+', text):
        for length in range(2, min(5, len(seq) + 1)):
            for i in range(len(seq) - length + 1):
                words.append(seq[i:i+length])
    return words


def _tokenize_joined(text: str) -> str:
    return ' '.join(_tokenize(text))



# ═══════════════════════════════════════════════════════════════
# BM25 — 稀疏检索
# ═══════════════════════════════════════════════════════════════

class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(d.split()) for d in documents) / len(documents) if documents else 1.0
        if self.avgdl == 0:
            self.avgdl = 1.0
        self.doc_freqs = self._df()
        self.idf = self._idf()

    def _df(self) -> Dict[str, int]:
        freq = {}
        for doc in self.documents:
            for word in set(doc.split()):
                freq[word] = freq.get(word, 0) + 1
        return freq

    def _idf(self) -> Dict[str, float]:
        N = len(self.documents)
        return {w: math.log((N - df + 0.5) / (df + 0.5) + 1)
                for w, df in self.doc_freqs.items()}

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        qwords = query.split()
        scores = []
        for idx, doc in enumerate(self.documents):
            s = self._score(doc, qwords)
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score(self, doc: str, qwords: List[str]) -> float:
        words = doc.split()
        dl = len(words)
        score = 0.0
        for w in qwords:
            if w not in self.idf:
                continue
            tf = words.count(w)
            if tf == 0:
                continue
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[w] * num / den
        return score


# ═══════════════════════════════════════════════════════════════
# RRF
# ═══════════════════════════════════════════════════════════════

class RRF:
    @staticmethod
    def fuse(results_list: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for results in results_list:
            for rank, (doc_idx, orig_score) in enumerate(results):
                rrf = 1.0 / (k + rank + 1)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + rrf
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════
# EmbeddingIndex — 稠密语义检索（智谱 Embedding API + numpy）
# ═══════════════════════════════════════════════════════════════

class EmbeddingIndex:

    def __init__(self):
        self._ids: List[str] = []
        self._embeddings = None  # np.ndarray (n, d)
        self._ready = False

    def _call_api(self, texts: list) -> list:
        """调用智谱 Embedding API，返回 embedding 列表"""
        if not ZHIPU_API_KEY:
            return []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ZHIPU_API_KEY}"
        }
        payload = {
            "model": ZHIPU_EMBEDDING_MODEL,
            "input": texts
        }
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(ZHIPU_EMBEDDING_ENDPOINT, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
            return [item["embedding"] for item in sorted(result.get("data", []), key=lambda x: x.get("index", 0))]
        except Exception as e:
            print(f"[WARN] Embedding API call failed: {e}")
            return []

    def populate_from_precomputed(self, ids: List[str], embeddings: List[List[float]]):
        """从预计算向量灌入 numpy 数组"""
        self._ids = list(ids)
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._ready = True
        print(f"[OK] Loaded {len(self._ids)} pre-computed embeddings (dim={self._embeddings.shape[1]})")

    def encode_documents_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量编码文档，返回向量列表（供 build_data.py 使用）"""
        print(f"[INFO] Encoding {len(texts)} documents via API ...")
        t0 = time.time()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch)
            if embeddings:
                all_embeddings.extend(embeddings)
            if (i // batch_size + 1) % 5 == 0:
                print(f"  ... {min(i + batch_size, len(texts))}/{len(texts)}")
        print(f"[OK] Encoding done in {time.time() - t0:.1f}s ({len(all_embeddings)} vectors)")
        return all_embeddings

    def encode_query(self, query: str) -> Optional[List[float]]:
        result = self._call_api([query])
        return result[0] if result else None

    def encode_text(self, text: str) -> Optional[List[float]]:
        return self.encode_query(text[:2000])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._ready or self._embeddings is None or len(self._ids) == 0:
            return []
        qvec = self.encode_query(query)
        if qvec is None:
            return []
        qvec = np.array(qvec, dtype=np.float32)
        # 余弦相似度（embeddings 已归一化）
        sims = np.dot(self._embeddings, qvec)
        indices = np.argsort(sims)[::-1][:top_k * 2]
        results = []
        for i in indices:
            sim = float(sims[i])
            if sim > 0.3:
                try:
                    idx = int(self._ids[i])
                    results.append((idx, sim))
                except ValueError:
                    continue
        return results[:top_k]

    def is_ready(self) -> bool:
        return self._ready and len(self._ids) > 0


# ═══════════════════════════════════════════════════════════════
# SimpleSearchEngine — 混合检索引擎
# ═══════════════════════════════════════════════════════════════

class SimpleSearchEngine:
    def __init__(self):
        self.documents: List[Document] = []
        self.parent_docs: List[Document] = []
        self.child_docs: List[Document] = []
        self.parent_map: Dict[str, Document] = {}
        self.child_map: Dict[str, Document] = {}

        self.child_bm25: Optional[BM25] = None
        self.parent_bm25: Optional[BM25] = None

        self.embedding_index = EmbeddingIndex()

        self.child_order: Dict[str, List[str]] = {}

    def add_documents(self, docs: List[Document]):
        for doc in docs:
            self.documents.append(doc)
            if doc.chunk_type == "parent":
                self.parent_docs.append(doc)
                self.parent_map[doc.id] = doc
            elif doc.chunk_type == "child":
                self.child_docs.append(doc)
                self.child_map[doc.id] = doc
                if doc.parent_id not in self.child_order:
                    self.child_order[doc.parent_id] = []
                self.child_order[doc.parent_id].append(doc.id)

        if self.child_docs:
            self.child_bm25 = BM25([_tokenize_joined(d.content) for d in self.child_docs])
        if self.parent_docs:
            self.parent_bm25 = BM25([_tokenize_joined(d.content) for d in self.parent_docs])

    def _hybrid_search(self, query: str, docs: List[Document],
                       bm25_idx: Optional[BM25], top_k: int) -> List[Tuple[int, float]]:
        bm25_query = _tokenize_joined(query)
        result_lists: List[List[Tuple[int, float]]] = []

        if bm25_idx:
            sparse = bm25_idx.search(bm25_query, top_k * 2)
            if sparse:
                result_lists.append(sparse)

        if self.embedding_index.is_ready() and docs == self.child_docs:
            dense = self.embedding_index.search(query, top_k * 2)
            if dense:
                result_lists.append(dense)

        if len(result_lists) >= 2:
            return RRF.fuse(result_lists)
        elif len(result_lists) == 1:
            return result_lists[0]
        return []

    def _get_neighbors(self, parent_id: str, child_id: str,
                       n: int = 1) -> List[str]:
        ordered = self.child_order.get(parent_id, [])
        if child_id not in ordered:
            return []
        idx = ordered.index(child_id)
        start = max(0, idx - n)
        end = min(len(ordered), idx + n + 1)
        return [ordered[i] for i in range(start, end) if ordered[i] != child_id]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        child_fused = self._hybrid_search(query, self.child_docs, self.child_bm25, top_k * 3)

        parent_groups: Dict[str, List[Tuple[int, float]]] = {}
        for child_idx, score in child_fused:
            child = self.child_docs[child_idx]
            pid = child.parent_id
            if pid not in parent_groups:
                parent_groups[pid] = []
            parent_groups[pid].append((child_idx, score))

        parent_only = [d for d in self.parent_docs
                       if not any(c.parent_id == d.id for c in self.child_docs)]
        if parent_only:
            parent_only_texts = [d.content for d in parent_only]
            parent_only_bm25 = BM25(parent_only_texts)
            parent_fused = self._hybrid_search(query, parent_only, parent_only_bm25, top_k)
            for p_idx, score in parent_fused:
                pid = parent_only[p_idx].id
                if pid not in parent_groups:
                    parent_groups[pid] = []
                parent_groups[pid].append((-1, score))

        ranked = [(pid, max(s for _, s in children))
                  for pid, children in parent_groups.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_parents = set()
        seen_contents = set()

        for pid, total_score in ranked:
            if pid in seen_parents:
                continue
            seen_parents.add(pid)

            parent_doc = self.parent_map.get(pid)
            if not parent_doc:
                continue

            children = parent_groups[pid]
            children.sort(key=lambda x: x[1], reverse=True)

            child_results = []
            neighbor_ids = set()

            for c_idx, c_score in children[:3]:
                if c_idx == -1:
                    snippet = parent_doc.content[:500]
                    if snippet not in seen_contents:
                        seen_contents.add(snippet)
                        child_results.append({
                            "id": parent_doc.id,
                            "content": snippet,
                            "score": c_score,
                            "is_context": False
                        })
                else:
                    child_doc = self.child_docs[c_idx]
                    if child_doc.content not in seen_contents:
                        seen_contents.add(child_doc.content)
                        child_results.append({
                            "id": child_doc.id,
                            "content": child_doc.content,
                            "score": c_score,
                            "is_context": False
                        })
                    for nid in self._get_neighbors(pid, child_doc.id):
                        neighbor_ids.add(nid)

            for nid in neighbor_ids:
                cd = self.child_map.get(nid)
                if cd and cd.content not in seen_contents:
                    seen_contents.add(cd.content)
                    child_results.append({
                        "id": cd.id,
                        "content": cd.content,
                        "score": 0.0,
                        "is_context": True
                    })

            results.append({
                "parent": {
                    "id": parent_doc.id,
                    "content": parent_doc.content[:3000],
                    "category": parent_doc.category,
                    "score": float(total_score)
                },
                "children": child_results
            })

            if len(results) >= top_k:
                break

        return results

    def retrieve(self, query: str, top_k: int = 5, children_per_parent: int = 3) -> List[Dict]:
        return self.search(query, top_k)

    def get_stats(self) -> Dict:
        return {
            "parent_count": len(self.parent_docs),
            "child_count": len(self.child_docs),
            "jieba": HAS_JIEBA,
            "embedding_model": EMBEDDING_MODEL_NAME if self.embedding_index.is_ready() else None,
        }

    def is_available(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════
# RAGEngine 封装
# ═══════════════════════════════════════════════════════════════

class RAGEngine:
    def __init__(self):
        self.search_engine = SimpleSearchEngine()

    def add_documents(self, docs: List[Document]):
        self.search_engine.add_documents(docs)

    def retrieve(self, query: str, top_k: int = 5, children_per_parent: int = 3) -> List[Dict]:
        return self.search_engine.search(query, top_k)

    def build_from_txt_files(self, txt_files: List[str], encoding: str = 'gb18030'):
        docs = []
        doc_id = 0
        for txt_file in txt_files:
            if not os.path.exists(txt_file):
                print(f"[WARN] File not found: {txt_file}")
                continue
            try:
                with open(txt_file, 'r', encoding=encoding) as f:
                    content = f.read()
            except (UnicodeDecodeError, LookupError):
                print(f"[WARN] Encoding {encoding} failed for {txt_file}, trying utf-8")
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            file_category = os.path.basename(txt_file)
            if file_category.endswith('.txt'):
                file_category = file_category[:-4]
            laws = self._parse_laws(content, txt_file, file_category)
            if not laws:
                laws = [{'name': file_category, 'content': content, 'source_file': txt_file}]
            for law in laws:
                parent_id = f"doc_{doc_id}"
                docs.append(Document(
                    id=parent_id, content=law['content'][:3000],
                    chunk_type="parent", category=law['name']
                ))
                doc_id += 1
                for ch_name, ch_content in self._split_chapters(law['content'], law['name']):
                    if len(ch_content) > 100:
                        docs.append(Document(
                            id=f"doc_{doc_id}", content=ch_content,
                            chunk_type="child", parent_id=parent_id, category=law['name']
                        ))
                        doc_id += 1
                n_children = len([d for d in docs if d.parent_id == parent_id])
                print(f"[OK] {law['name']} -> 1 parent + {n_children} children")
        self.add_documents(docs)
        return docs

    def _parse_laws(self, content: str, source_file: str, file_category: str) -> list:
        laws = []
        pattern = r'（[一二三四五六七八九十]+）《([^》]+)》'
        matches = list(re.finditer(pattern, content))
        if not matches:
            alt = list(re.finditer(r'一、《([^》]+)》', content))
            if alt:
                return [{'name': '深圳市住房公积金管理办法', 'content': content, 'source_file': source_file}]
            return []
        for i, m in enumerate(matches):
            name = m.group(1) + '》'
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end]
            text = re.sub(r'（[一二三四五六七八九十]+）《[^》]+》', '', text)
            laws.append({'name': name, 'content': text.strip(), 'source_file': source_file})
        return laws

    def _split_chapters(self, content: str, law_name: str) -> list:
        chapters = []
        pattern = r'(第[一二三四五六七八九十百\d]+章[^\n]*)\n(.*?)(?=\n第[一二三四五六七八九十百\d]+章|$)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if matches and len(matches) >= 2:
            for m in matches:
                title = m.group(1).strip()
                body = m.group(2).strip()
                if len(body) > 100:
                    chapters.append((title, body))
            if chapters:
                return chapters
        paras = [p.strip() for p in content.split('\n') if p.strip()]
        for i in range(0, len(paras), 15):
            chunk = '\n'.join(paras[i:i+15])
            if len(chunk) > 100:
                chapters.append((f"第{i//15 + 1}部分", chunk))
        return chapters

    def rebuild_index(self, txt_files: List[str], encoding: str = 'gb18030'):
        self.search_engine = SimpleSearchEngine()
        self.build_from_txt_files(txt_files, encoding)

    def get_stats(self) -> Dict:
        return self.search_engine.get_stats()

    def is_available(self) -> bool:
        return True
