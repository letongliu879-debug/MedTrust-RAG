"""检索质量评测

指标：Recall@K, Precision@K, MRR, Hit@K, NDCG@K

面试话术：
"评测设计的关键是 ground truth 的定义——Huatuo-Lite 的每个 QA 对自身
就是一个 ground truth chunk。对于测试问题 q_i，我用它原文的 QA 对 ID
作为 relevant item，然后看检索系统能不能在 top-K 里找到它。

这比用人工标注做 relevance 更客观，因为避免了标注者主观判断。
代价是只衡量了'能否找回原 QA 对'，没衡量'找到的其他 QA 对是否也有用'。
这是检索评测里 precision/recall trade-off 的经典权衡。"
"""

import hashlib
import math
from collections import defaultdict

from src.utils.logger import logger


class RetrievalEvaluator:
    """检索质量评测器"""

    def __init__(self):
        self._k_values = [1, 3, 5, 10]

    def evaluate(
        self,
        test_pairs: list[dict],
        retrieve_fn,
    ) -> dict:
        """
        跑检索评测。

        Args:
            test_pairs: [{question, answer, id, ...}, ...]
            retrieve_fn: (query: str) -> list[dict]
                返回 [{text, metadata: {source_id}}, ...]

        Returns:
            {recall@1: 0.XX, mrr: 0.XX, ndcg@5: 0.XX, ...}
        """
        metrics = defaultdict(list)

        for i, pair in enumerate(test_pairs):
            query = pair["question"]
            target_id = pair["id"]

            results = retrieve_fn(query)
            retrieved_ids = [
                r.get("metadata", {}).get("source_id", -1)
                for r in results
            ]

            for k in self._k_values:
                top_k_ids = retrieved_ids[:k]
                # recall@k: target 在 top-k 里？
                hit = 1 if target_id in top_k_ids else 0
                metrics[f"hit@{k}"].append(hit)
                # precision@k: top-k 里 target 占多少
                metrics[f"precision@{k}"].append(hit / k)

            # MRR: 1/rank
            rank = self._find_rank(target_id, retrieved_ids)
            metrics["mrr"].append(1.0 / rank if rank > 0 else 0.0)

            # NDCG@5
            ndcg = self._ndcg_at_k(target_id, retrieved_ids, 5)
            metrics["ndcg@5"].append(ndcg)

            if (i + 1) % 50 == 0:
                logger.info(f"检索评测进度: {i + 1}/{len(test_pairs)}")

        # 汇总
        summary = {}
        for metric_name, values in sorted(metrics.items()):
            summary[metric_name] = round(sum(values) / len(values), 4)

        summary["total_queries"] = len(test_pairs)
        logger.info(f"检索评测完成: {summary}")
        return summary

    def evaluate_with_hybrid(
        self,
        test_pairs: list[dict],
        hybrid_retriever,
        collection_name: str,
    ) -> dict:
        """用混合检索器评测"""

        def retrieve_fn(query):
            results = hybrid_retriever.retrieve(
                query=query,
                collection_name=collection_name,
                top_k=10,
            )
            return results

        return self.evaluate(test_pairs, retrieve_fn)

    # ============ 辅助 ============

    @staticmethod
    def _find_rank(target_id: int, retrieved_ids: list[int]) -> int:
        try:
            return retrieved_ids.index(target_id) + 1
        except ValueError:
            return 0

    @staticmethod
    def _ndcg_at_k(target_id: int, retrieved_ids: list[int], k: int) -> float:
        """NDCG@K：target 在 rank 位置则得分 1/log2(rank+1)"""
        rank = RetrievalEvaluator._find_rank(target_id, retrieved_ids)
        if rank == 0 or rank > k:
            return 0.0
        dcg = 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)  # 最优：rank=1
        return dcg / idcg
