"""证据收集 Agent

职责：将用户问题分解为多路子查询 → 混合检索 → 去重 → reranker 精排 → 输出 EvidenceBundle

优化要点:
- 合并 classify + subquery 为一次 LLM 调用（减少 API 往返）
- 简单查询（单实体）用关键词规则跳过 LLM，直接单路检索
"""

import asyncio

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.agents.base import BaseAgent, EvidenceBundle, AgentResult
from src.rag.hybrid_retriever import hybrid_retriever
from src.rag.reranker import reranker
from src.llm.model_factory import ModelFactory
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.trace import get_trace
from src.utils.exec_log import get_log


class RetrieverAgent(BaseAgent):
    """证据收集 Agent"""

    def __init__(self):
        super().__init__("retriever")
        self._collection_name = config.get("dataset.collection_name", "med_all")

    @traceable(
        run_type="chain",
        name="RetrieverAgent",
        metadata={"stage": "agent_retrieve"},
    )
    async def run(
        self,
        query: str,
        department: str = None,
        top_k: int = None,
    ) -> AgentResult:
        """
        收集证据。
        """
        k = top_k or config.get("rag.retrieval_top_k", 5)

        # 1. 确保 BM25 索引就绪
        hybrid_retriever._ensure_bm25_index(self._collection_name)

        # 2. 判断检索策略：简单问题单路检索，多跳问题拆子查询
        trace_ctx = get_trace()
        if self._is_simple_query(query):
            # 简单查询：跳过 LLM，直接单路检索
            is_multi = False
            sub_queries = [query]
            logger.info(f"简单查询，单路检索: '{query[:30]}...'")
        else:
            # 一次 LLM 调用同时完成「判断多跳」+「生成子查询」
            if trace_ctx:
                with trace_ctx.step("1a_subquery_gen") as ts:
                    is_multi, sub_queries = await self._classify_and_generate(query)
                    ts.metadata["is_multi_hop"] = is_multi
                    ts.metadata["sub_query_count"] = len(sub_queries)
            else:
                is_multi, sub_queries = await self._classify_and_generate(query)

        # 3. 检索（多跳并行，单路直接）
        all_results = []
        _cm = None
        _step = None
        if trace_ctx:
            _cm = trace_ctx.step("1b_retrieval")
            _step = _cm.__enter__()
        try:
            if is_multi and len(sub_queries) > 1:
                from functools import partial
                tasks = [
                    asyncio.to_thread(
                        partial(
                            hybrid_retriever.retrieve,
                            collection_name=self._collection_name,
                            query=sq,
                            top_k=10,
                            department=department,
                        )
                    )
                    for sq in sub_queries
                ]
                batch_results = await asyncio.gather(*tasks)
                for results in batch_results:
                    all_results.extend(results)
                logger.info(f"多跳问题，{len(sub_queries)} 路子查询并行检索，共 {len(all_results)} 条")
            else:
                all_results = hybrid_retriever.retrieve(
                    collection_name=self._collection_name,
                    query=query,
                    top_k=15,
                    department=department,
                )
        finally:
            if _cm is not None:
                _cm.__exit__(None, None, None)
        if _step is not None:
            _step.metadata["raw_results"] = len(all_results)

        # 4. 去重（全文 hash）
        import hashlib
        seen = set()
        unique = []
        for r in all_results:
            key = hashlib.md5(r["text"].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(r)

        unique.sort(key=lambda r: r.get("score", 0), reverse=True)
        unique = unique[:15]

        # 5. Reranker 精排
        if unique and config.get("rag.reranker.enabled", True):
            if trace_ctx:
                with trace_ctx.step("1c_rerank") as ts:
                    ts.metadata["candidates"] = len(unique)
                    reranked = reranker.rerank(query, unique, top_k=k)
                    final_chunks = reranked if reranked else unique[:k]
                    ts.metadata["final"] = len(final_chunks)
            else:
                reranked = reranker.rerank(query, unique, top_k=k)
                final_chunks = reranked if reranked else unique[:k]
        else:
            final_chunks = unique[:k]

        # 6. 检索质量门控：精排后 top rerank 分数过低 → 判定检索失败
        # 仅在 reranker 已启用且实际打过分时检查（reranker 禁用时 chunks 无 rerank_score）
        min_quality = config.get("rag.reranker.min_quality_score", 0.1)
        has_rerank_scores = final_chunks and "rerank_score" in final_chunks[0]
        if has_rerank_scores and final_chunks:
            top_score = final_chunks[0].get("rerank_score", 0)
            if top_score < min_quality:
                logger.warning(
                    f"检索质量不足: top_rerank={top_score:.4f} < min={min_quality}, "
                    f"query='{query[:30]}...', 跳过 LLM 生成"
                )
                return AgentResult(
                    answer="",
                    citations=[],
                    confidence=0.0,
                )
        else:
            # reranker 过滤后无结果
            logger.warning(f"精排后无结果: query='{query[:30]}...'")
            return AgentResult(answer="", citations=[], confidence=0.0)

        logger.info(
            f"RetrieverAgent: query='{query[:30]}...', "
            f"sub_queries={len(sub_queries)}, "
            f"candidates={len(unique)}, final={len(final_chunks)}"
        )

        # LangSmith trace
        rt = get_current_run_tree()
        if rt:
            final_scores = [
                {"rank": i + 1, "rerank_score": c.get("rerank_score"),
                 "rrf_score": c.get("score"), "text_head": c["text"][:60]}
                for i, c in enumerate(final_chunks)
            ]
            rt.add_metadata({
                "sub_query_count": len(sub_queries),
                "candidates_before_dedup": len(all_results),
                "candidates_after_dedup": len(unique),
                "final_after_rerank": len(final_chunks),
            })
            rt.extra = {
                "sub_queries": sub_queries,
                "final_scores": final_scores,
            }

        return AgentResult(
            answer="",
            citations=final_chunks,
            confidence=self._avg_score(final_chunks),
        )

    # ============ 查询分类 + 子查询生成（合并为一次 LLM 调用）============

    async def _classify_and_generate(self, query: str) -> tuple[bool, list[str]]:
        """
        一次 LLM 调用同时完成：
        1. 判断是否多跳问题
        2. 如果是，生成多路子查询

        合并后从 2 次 API 往返 → 1 次，省 ~1-2s 延迟。
        """
        prompt = (
            "你是一个医疗检索专家。请判断以下医学问题是否需要多角度检索，"
            "如果需要，请生成 2-4 个不同角度的检索子查询。\n\n"
            "## 判断标准\n"
            "- 多跳：问题同时涉及多个独立医疗实体（如：一种疾病+另一种疾病、疾病+具体药物名、疾病+具体症状）\n"
            "- 单跳：仅问一种疾病的病因/治疗/预防/症状，或一个药名的副作用\n\n"
            "## 用户问题\n"
            f"{query}\n\n"
            "## 输出 JSON\n"
            "{{\n"
            '  "is_multi_hop": true/false,\n'
            '  "sub_queries": ["子查询1", "子查询2", ...]\n'
            "}}\n\n"
            "## 规则\n"
            "- 子查询应覆盖不同维度（症状分析、治疗方案、药物信息、科室匹配等）\n"
            "- 每个子查询简洁明确，10-30 字\n"
            "- 必须保留原问题中的具体医学实体（疾病名、药名、症状名），不能泛化\n"
            "- 只输出 JSON"
        )
        try:
            llm = ModelFactory.create_chat_model()
            result = await llm.ainvoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)

            from src.llm.chains import parse_json_from_text
            parsed = parse_json_from_text(content)

            # exec_log: 记录子查询生成
            elog = get_log()
            if elog:
                elog.step("subquery_gen",
                    input={"prompt": prompt},
                    raw=content,
                    output={"is_multi_hop": parsed.get("is_multi_hop"), "sub_queries": parsed.get("sub_queries")},
                )

            is_multi = parsed.get("is_multi_hop", False)
            sub_queries = parsed.get("sub_queries", [])

            if is_multi and isinstance(sub_queries, list) and len(sub_queries) > 0:
                sub_queries = [s for s in sub_queries if isinstance(s, str) and s.strip()]
                if sub_queries:
                    if query not in sub_queries:
                        sub_queries.insert(0, query)
                    return True, sub_queries[:5]

            return False, [query]

        except TimeoutError:
            logger.warning(f"子查询生成 LLM 超时（60s），降级为规则分解")
            return False, [query]
        except Exception as e:
            logger.warning(f"分类+子查询生成失败，降级单路检索: {e}")
            return False, [query]

    # ============ 简单查询快速判断（不调 LLM）============

    @staticmethod
    def _is_simple_query(query: str) -> bool:
        """
        关键词规则判断是否为简单查询。

        简单查询特征：
        - 问单一疾病的单一维度（病因/治疗/症状/预防/药物）
        - 不含药物名+疾病名、不含多种疾病
        - 不含比较词（"和"、"与"、"对比"、"区别"）

        满足条件直接跳过多跳判断 LLM 调用。
        """
        # 多实体特征词：同时出现疾病+药物，或多种疾病
        multi_entity_patterns = [
            "和", "与", "以及", "对比", "区别", "比较",
            "同时", "合并", "伴有", "伴随", "还有",
        ]
        # 简单问题模式词（问单一维度的）
        # 注意："药"太泛，放在最后且只匹配明确提问药物的问题
        simple_patterns = [
            "是什么", "什么是", "怎么治", "如何治疗", "治疗方法",
            "症状", "表现", "病因", "原因", "预防", "注意事项",
            "饮食", "护理", "诊断", "检查",
            "吃什么药", "用什么药", "吃啥药", "药物推荐",
        ]

        has_multi_connector = any(p in query for p in multi_entity_patterns)
        has_simple_pattern = any(p in query for p in simple_patterns)

        # 有简单模式且没有多实体连接词 → 大概率是简单查询
        if has_simple_pattern and not has_multi_connector:
            # 额外检查：问题长度不太长（复杂问题通常较长）
            if len(query) < 40:
                return True

        return False

    @staticmethod
    def _rule_based_subqueries(query: str) -> list[str]:
        """固定规则降级"""
        import re
        clean = re.sub(r"[？?，,。！!]", " ", query).strip()
        parts = [q.strip() for q in clean.split() if len(q.strip()) > 2]
        return [query] + (parts[:2] if parts else [])

    @staticmethod
    def _avg_score(chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        scores = [c.get("rerank_score") or c.get("score", 0) for c in chunks]
        return round(sum(scores) / len(scores), 3)
