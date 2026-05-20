# -*- coding: utf-8 -*-
"""
增强版RAG系统（Vercel 版 — 无服务端持久化）
- 先子后父检索 + BM25 + BGE 混合检索 + RRF 融合
- Query 扩展 + MMR 多样性选择 + LLM Re-ranking
"""

import json
import os
import urllib.request
import urllib.error
from typing import List, Dict

from memory import UserMemory

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

try:
    from agent import InsuranceAgent
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False


class EnhancedRAG:
    """增强版RAG系统"""

    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL
        self.model = DEEPSEEK_MODEL

        self.rag_engine = None
        try:
            from rag_engine import RAGEngine as _RAGEngine
            self.rag_engine = _RAGEngine()

            stats = self.rag_engine.get_stats()
            total_docs = stats["parent_count"] + stats["child_count"]

            if total_docs == 0:
                print("[INFO] Building index from source files...")
                source_files = [
                    "深圳市社保法律法规汇编.txt",
                    "深圳市住房公积金法律法规汇编.txt",
                ]
                self.rag_engine.rebuild_index(source_files, encoding='gb18030')
                stats = self.rag_engine.get_stats()
                total_docs = stats["parent_count"] + stats["child_count"]

            if self.rag_engine.is_available():
                print(f"[OK] RAG engine ready ({total_docs} docs)")
            else:
                print("[WARN] RAG engine not available")
        except Exception as e:
            print(f"[WARN] RAG engine init failed: {e}")
            self.rag_engine = None

    # ── DeepSeek API ───────────────────────────────────────────

    def _call_deepseek_api(self, messages: List[Dict], temperature: float = 0.3,
                           max_tokens: int = 2048, stream: bool = True, callback=None) -> str:
        if not self.api_key:
            msg = "系统错误：API 密钥未设置，请联系管理员"
            if callback:
                for char in msg:
                    callback(char)
            return msg

        try:
            url = f"{self.base_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')

            if stream and callback:
                return self._parse_stream(req, callback)
            else:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            return f"HTTP error {e.code}: {error_body[:500]}"
        except urllib.error.URLError as e:
            return f"Network error: {e.reason}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _parse_stream(self, req, callback) -> str:
        full_content = ""
        line_buffer = ""
        bytes_buf = bytearray()
        with urllib.request.urlopen(req, timeout=120) as response:
            while True:
                chunk = response.read(2048)
                if not chunk:
                    break
                bytes_buf.extend(chunk)

                try:
                    text = bytes_buf.decode('utf-8')
                    bytes_buf.clear()
                except UnicodeDecodeError as e:
                    if e.reason == 'unexpected end of data':
                        text = bytes_buf[:e.start].decode('utf-8')
                        del bytes_buf[:e.start]
                    else:
                        text = bytes_buf.decode('utf-8', errors='replace')
                        bytes_buf.clear()

                line_buffer += text
                while '\n' in line_buffer:
                    line, line_buffer = line_buffer.split('\n', 1)
                    line = line.strip()
                    if not line or line == 'data: [DONE]':
                        continue
                    if not line.startswith('data: '):
                        continue
                    try:
                        data_json = json.loads(line[6:].strip())
                        delta = data_json.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            full_content += content
                            callback(content)
                    except (json.JSONDecodeError, KeyError):
                        continue

            if bytes_buf:
                line_buffer += bytes_buf.decode('utf-8', errors='replace')
                bytes_buf.clear()

        return full_content

    # ── Query 扩展 ──────────────────────────────────────────────

    def _expand_query(self, query: str) -> List[str]:
        if not self.api_key:
            return [query]

        prompt = f"""将以下用户问题改写为 2-3 个不同的表述，用于提高法律条文检索的召回率。

【原始问题】
{query}

【要求】
- 保持原意不变
- 使用不同的措辞、同义词、正式/口语化变体
- 每行一个表述
- 只输出改写结果，不要编号、不要解释"""

        messages = [
            {"role": "system", "content": "你是查询改写助手，只输出改写后的查询，每行一个。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_deepseek_api(
                messages, temperature=0.7, max_tokens=200,
                stream=False, callback=None
            )
            variants = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and len(line) >= 4:
                    variants.append(line)
            if variants:
                return [query] + variants[:3]
        except Exception:
            pass
        return [query]

    # ── LLM Re-ranking ──────────────────────────────────────────

    def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        if len(docs) <= 3 or not self.api_key:
            return docs

        doc_texts = []
        for i, doc in enumerate(docs):
            p = doc["parent"]
            matched = [c["content"][:200] for c in doc["children"] if not c.get("is_context")]
            snippets = " | ".join(matched) if matched else p["content"][:200]
            doc_texts.append(f"[{i}] {p['category']}: {snippets}")

        prompt = f"""评估以下文档与用户问题的相关度，按相关度从高到低排序，只输出排序后的序号列表。

【用户问题】
{query}

【候选文档】
{chr(10).join(doc_texts)}

只输出序号列表，如: 2, 0, 1, 3"""

        messages = [
            {"role": "system", "content": "你是检索排序助手，只输出序号列表。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_deepseek_api(
                messages, temperature=0.1, max_tokens=50,
                stream=False, callback=None
            )
            import re
            numbers = re.findall(r'\d+', response)
            new_order = []
            seen = set()
            for n in numbers:
                idx = int(n)
                if 0 <= idx < len(docs) and idx not in seen:
                    new_order.append(idx)
                    seen.add(idx)
            if len(new_order) >= 2:
                return [docs[i] for i in new_order]
        except Exception:
            pass
        return docs

    # ── MMR 多样性选择 ──────────────────────────────────────────

    def _document_similarity(self, doc_a: Dict, doc_b: Dict) -> float:
        """Jaccard similarity — API embedding 对 MMR 来说太慢"""
        text_a = doc_a["parent"]["content"]
        text_b = doc_b["parent"]["content"]
        from rag_engine import _tokenize
        wa = set(_tokenize(text_a))
        wb = set(_tokenize(text_b))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def _mmr_select(self, query: str, candidates: List[Dict],
                    top_k: int, lam: float = 0.85) -> List[Dict]:
        if len(candidates) <= top_k:
            return candidates

        remaining = list(candidates)
        selected = []
        remaining.sort(key=lambda r: r["parent"]["score"], reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float('inf')
            for i, cand in enumerate(remaining):
                relevance = cand["parent"]["score"]
                max_sim = max(self._document_similarity(cand, sel) for sel in selected)
                mmr = lam * relevance - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    # ── Query 上下文整合 ──────────────────────────────────────────

    def _contextual_rewrite(self, query: str, history: List[Dict], user_summary: str) -> str:
        if not history or not self.api_key:
            return query

        user_history = [h for h in history if h['role'] == 'user']
        history_text = "\n".join(f"用户: {h['content']}" for h in user_history)

        prompt = f"""根据对话历史和用户画像，将用户的追问改写为一个完整、自包含的问题。

【对话历史】
{history_text}

【用户画像】
{user_summary}

【当前追问】
{query}

【要求】
- 如果追问引用了对话历史中的内容（如"为什么"、"那XX呢"、"具体多少钱"），必须结合历史补全为完整问题
- 保持原意不变
- 如果追问已经是完整的独立问题，直接返回原句
- 只输出改写后的问题，不要任何解释"""

        messages = [
            {"role": "system", "content": "你是查询改写助手，将模糊追问改写为完整问题。只输出改写结果，不要解释。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._call_deepseek_api(
                messages, temperature=0.3, max_tokens=120,
                stream=False, callback=None
            )
            if response and len(response.strip()) >= 4:
                return response.strip()
        except Exception:
            pass
        return query

    # ── 检索逻辑 ──────────────────────────────────────────────────

    def _add_citation(self, answer: str, relevant_docs: List[Dict]) -> str:
        if not relevant_docs or "信息来源" in answer:
            return answer
        law_names = []
        for doc in relevant_docs[:3]:
            name = doc.get('category', '')
            if name and name not in law_names:
                law_names.append(name)
        if law_names:
            answer += "\n\n**信息来源**：" + " | ".join(law_names)
        return answer

    def chat_streaming(self, query: str, callback, history: List[Dict] = None,
                       profile: Dict = None):
        """流式聊天接口。profile 由前端传入，包含用户画像字段。"""
        user_memory = UserMemory.from_dict(profile or {})
        user_summary = user_memory.get_summary()

        # 0.5. Query 上下文整合
        rewritten_query = self._contextual_rewrite(query, history or [], user_summary)

        # 1. Agent 处理
        if HAS_AGENT:
            agent = InsuranceAgent(user_memory)
            agent_result, need_more = agent.process(rewritten_query)
            if agent_result:
                for char in agent_result:
                    callback(char)
                return agent_result, need_more

        # 2. RAG 检索
        if self.rag_engine and self.rag_engine.is_available():
            raw_results = self.rag_engine.retrieve(rewritten_query, top_k=7, children_per_parent=2)
            all_results = []
            for r in raw_results:
                children_sorted = sorted(r["children"], key=lambda c: c["score"], reverse=True)[:2]
                all_results.append({
                    "parent": r["parent"],
                    "children": children_sorted
                })
            all_results.sort(key=lambda r: r["parent"]["score"], reverse=True)

            if len(all_results) >= 7:
                all_results = self._mmr_select(rewritten_query, all_results, top_k=7, lam=0.85)

            candidates = all_results[:7]
            if len(candidates) > 5:
                candidates = self._rerank(rewritten_query, candidates)
            top_results = candidates[:3]

            context_parts = []
            relevant_docs = []

            for r in top_results:
                parent = r["parent"]
                if parent["score"] <= 0:
                    continue
                context_parts.append(f"【{parent['category']}】(相关度:{parent['score']:.2f})")
                context_parts.append(parent["content"])
                relevant_docs.append({"category": parent["category"], "content": parent["content"]})

                matched = [c for c in r["children"] if not c.get("is_context")]
                ctx = [c for c in r["children"] if c.get("is_context")]

                if matched:
                    context_parts.append("--- 匹配条款 ---")
                    for child in matched:
                        context_parts.append(f"* {child['content']}")
                if ctx:
                    context_parts.append("--- 上下文（相邻条款） ---")
                    for child in ctx:
                        context_parts.append(f"  {child['content']}")

            if context_parts:
                system_prompt = f"""你是深圳市五险一金智能咨询助手，只服务于深圳市参保群众。

【地域限制 - 最高优先级】
- **只回答深圳市五险一金相关问题**
- 遇到其他城市或其他领域问题，**必须拒绝**并回复："您好，我是深圳市五险一金智能咨询助手，只解答深圳市社保公积金相关问题。其他城市政策或其他问题建议咨询当地12333热线。"

【精准匹配原则】
1. **只引用知识库中存在的法规条款**，绝不编造
2. **每句话都要有据可查**，引用时使用 **信息来源**：[法规名称]
3. 如果参考信息不足以回答问题，明确告知用户

【危险边界 - 严格禁止】
- 禁止：编造法规条款、捏造数字
- 禁止：回答深圳以外城市的社保政策
- 禁止：回答与五险一金无关的问题
- 禁止：使用"大概"、"可能"、"一般来说"等模糊表述
- 禁止：给出知识库中没有的政策依据

【回答格式要求】
- 使用 Markdown 格式化回答
- 涉及政策的回答必须引用来源法规
- 数字、政策条款必须与知识库完全一致

【用户画像】
{user_summary}

参考信息：
{chr(10).join(context_parts)}"""

                user_content = ""
                if history:
                    user_history = [h for h in history if h['role'] == 'user']
                    if user_history:
                        history_text = "\n".join(f"用户: {h['content']}" for h in user_history)
                        user_content += f"【对话历史】\n{history_text}\n\n"
                user_content += f"【当前问题】\n{rewritten_query}\n\n请根据以上参考信息和对话历史回答。"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]

                full_content = self._call_deepseek_api(messages, temperature=0.3, callback=callback)
                answer = self._add_citation(full_content, relevant_docs)
                return answer if answer else "抱歉，我暂时无法回答这个问题。", False

        no_info_msg = "抱歉，我暂时没有找到相关的政策依据来回答这个问题。\n\n建议您拨打12333社保热线咨询，或登录深圳市社保局官网查询。"
        for char in no_info_msg:
            callback(char)
        return no_info_msg, False
