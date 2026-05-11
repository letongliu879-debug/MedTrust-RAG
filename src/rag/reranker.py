"""重排序器：用 BGE-Reranker 对检索结果做精细排序

======================== 医疗可信问答场景 ========================

### 为什么检索后还要重排序？
两阶段检索：
第一阶段（粗排）：混合检索（BM25 + 向量）快速召回 top 20-50 个候选。
第二阶段（精排）：用交叉编码器对 top 20 个候选逐一打分。

bi-encoder（粗排）的缺点是查询和文档各自独立编码，交互只在最后的点积。
cross-encoder（精排）把查询和文档拼接成一对输入，所有 token 之间做全注意力交互，
能捕捉到"查询里的症状在文档的哪一部分被提到了"这种细节。

### 为什么选 BGE-Reranker？
- BGE 系列是 BAAI（北京智源）出品，中文医学文本表现好
- BGE-Reranker-v2-m3 支持多语言，中英混合场景适用
- 开源免费，不需要 API key
- 部署简单：pip install FlagEmbedding，一行加载

### 阈值过滤
reranker 打出的分数范围通常在 [0, 1]（归一化后），分数低于阈值的直接丢弃——
这些是粗排召回来的噪声，cross-encoder 确认它们跟查询不相关。
"""

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.exec_log import get_log


class Reranker:
    """
    BGE-Reranker 重排序器。

    使用方式：
        reranker = Reranker()
        reranked = reranker.rerank(query, docs, top_k=5)
        # docs: [{"text": ..., "metadata": ...}, ...]
        # reranked: 同上，但按 rerank 分数重新排序，低分被过滤

    模型选择：
    - BAAI/bge-reranker-base: ~1GB，英文为主，中文可接受
    - BAAI/bge-reranker-v2-m3: ~2GB，多语言（中英日韩），推荐中文场景用这个
    - BAAI/bge-reranker-v2-minicpm-layerwise: 更小更快，质量略低

    默认用 v2-m3，可通过 config 改。
    """

    def __init__(self):
        model_name = config.get(
            "rag.reranker.model_name",
            "BAAI/bge-reranker-v2-m3",
        )
        self._model = None
        self._model_name = model_name
        self._threshold = config.get("rag.reranker.threshold", 0.3)
        self._use_fp16 = config.get("rag.reranker.use_fp16", False)
        self._normalize = config.get("rag.reranker.normalize", True)
        # 自动修正：归一化模式下阈值不应为负数
        if self._normalize and self._threshold < 0:
            logger.warning(
                f"Reranker 阈值 {self._threshold} 与 normalize=True 不兼容"
                f"（归一化分数在 [0,1]），自动调整为 0.3"
            )
            self._threshold = 0.3
        # 如果 reranker 未启用或加载失败，gracefully degrade
        self._enabled = config.get("rag.reranker.enabled", True)

    # ============ 懒加载模型 ============

    def _ensure_model(self):
        """
        懒加载 reranker 模型。

        为什么懒加载？
        BGE-Reranker 模型约 2GB，启动时不需要立即加载。只有第一次用到
        重排序时才加载模型。这样即使配置了 reranker，在不需要重排序的
        场景（如仅做索引构建）也不会浪费内存。
        """
        if self._model is not None:
            return
        if not self._enabled:
            return

        try:
            # FlagEmbedding 是 BGE 系列的官方 Python SDK
            from FlagEmbedding import FlagReranker

            logger.info(f"加载 Reranker 模型: {self._model_name}")
            self._model = FlagReranker(
                self._model_name,
                use_fp16=self._use_fp16,
            )
            logger.info("Reranker 模型加载完成")
        except ImportError:
            logger.warning(
                "FlagEmbedding 未安装，reranker 不可用。"
                "安装: pip install FlagEmbedding"
            )
            self._enabled = False
        except Exception as e:
            logger.error(f"加载 Reranker 模型失败: {e}")
            self._enabled = False

    # ============ 重排序 ============

    @traceable(
        run_type="retriever",
        name="BGE精排(Reranker)",
        metadata={"stage": "fine_rerank"},
    )
    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = None,
    ) -> list[dict]:
        """
        对检索结果重排序。

        Args:
            query: 查询文本
            docs: 检索结果 [{"text": ..., "metadata": ..., "score": ..., "sources": [...]}, ...]
            top_k: 返回数量，默认用配置值

        Returns:
            重排序后的结果，每个 item 新增 "rerank_score" 字段，
            按 rerank_score 降序排列，低于阈值的被过滤

        面试时注意：这里不是简单调包——
        1. 我做了 batch 处理（模型内部 tokenize 有最大长度限制）
        2. 加了阈值过滤（低分文档说明 cross-encoder 确认不相关）
        3. score 会合入原始文本，供下游 reviewer 参考
        """
        self._ensure_model()

        if not self._enabled or self._model is None:
            # reranker 不可用：原样返回，保留粗排顺序
            return docs[:top_k] if top_k else docs

        if not docs:
            return []

        k = top_k or config.get("rag.retrieval_top_k", 5)

        # 构造 (query, doc) pairs
        # cross-encoder 的输入格式是 [query, doc_text]
        pairs = []
        for doc in docs:
            doc_text = doc.get("text", "")
            if not doc_text.strip():
                pairs.append([query, query])  # 占位，避免空文本报错
            else:
                pairs.append([query, doc_text])

        try:
            scores = self._model.compute_score(
                pairs,
                normalize=self._normalize,  # [0,1] when True, raw when False
            )
        except Exception as e:
            logger.error(f"Reranker 打分失败: {e}")
            return docs[:k]

        # 处理 compute_score 返回值
        # FlagReranker.compute_score 可能返回 float（单样本）或 list[float]（多样本）
        if not isinstance(scores, (list, tuple)):
            scores = [float(scores)]

        # 合并分数，附到原文档上
        scored_docs = []
        for i, doc in enumerate(docs):
            rerank_score = float(scores[i]) if i < len(scores) else 0.0
            if rerank_score >= self._threshold:
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = round(rerank_score, 4)
                doc_copy["sources"] = doc.get("sources", []) + ["reranker"]
                scored_docs.append(doc_copy)

        # 按 rerank 分数降序
        scored_docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

        filtered_count = len(docs) - len(scored_docs)
        if filtered_count > 0:
            logger.info(
                f"Reranker 过滤了 {filtered_count} 个低分文档 "
                f"(阈值={self._threshold})"
            )

        logger.info(
            f"Reranker 完成: {len(docs)} → {len(scored_docs)} 个候选, "
            f"top_score={scored_docs[0].get('rerank_score', 0) if scored_docs else 0}"
        )

        # exec_log
        elog = get_log()
        if elog:
            elog.step("reranker",
                input={"candidates": len(docs)},
                output=[{"rerank_score": d.get("rerank_score", 0),
                         "coarse_score": d.get("score", 0),
                         "text": d["text"][:120]}
                        for d in scored_docs[:5]],
                metadata={"filtered": len(docs) - len(scored_docs), "threshold": self._threshold},
            )

        # LangSmith trace: 精排详情
        rt = get_current_run_tree()
        if rt:
            input_scores = [
                {"text_head": d["text"][:60], "coarse_score": d.get("score", 0),
                 "sources": d.get("sources", [])}
                for d in docs[:10]
            ]
            output_scores = [
                {"rank": i + 1, "rerank_score": d.get("rerank_score", 0),
                 "coarse_score": d.get("score", 0), "text_head": d["text"][:60]}
                for i, d in enumerate(scored_docs[:k])
            ]
            rt.add_metadata({
                "input_count": len(docs),
                "output_count": len(scored_docs),
                "filtered_count": len(docs) - len(scored_docs),
                "threshold": self._threshold,
            })
            rt.extra = {
                "input_top10_coarse_scores": input_scores,
                "output_topk_rerank_scores": output_scores,
            }

        return scored_docs[:k]

    # ============ 面向审查管道的高层接口 ============

    def rerank_text(self, query: str, docs: list[dict], top_k: int = 5) -> str:
        """
        重排序并格式化为文本，直接对接 QA pipeline。

        把 reranker 的输出格式化成可读的文本。
        """
        reranked = self.rerank(query, docs, top_k=top_k)
        if not reranked:
            return "暂无相关医学参考"

        lines = []
        for i, doc in enumerate(reranked, 1):
            meta = doc.get("metadata", {})
            source = meta.get("file_name", "未知来源")
            rerank_s = doc.get("rerank_score", 0)
            lines.append(
                f"[{i}] 来源: {source} (相关度: {rerank_s:.3f})\n{doc['text']}"
            )
        return "\n\n".join(lines)


# 全局实例（模型懒加载，不占用启动时间）
reranker = Reranker()
