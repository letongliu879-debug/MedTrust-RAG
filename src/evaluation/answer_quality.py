"""答案质量评测

指标：BLEU, ROUGE-L, BERTScore

面试话术：
"BLEU 和 ROUGE 衡量词汇层面的重叠，BERTScore 用 BERT 编码后
计算语义相似度。三者的互补性好——BLEU 对精确匹配敏感，
ROUGE 对召回敏感，BERTScore 对同义表达宽容。

对医疗 QA 场景，BERTScore 通常最接近人工评估结果，
因为同样的医学建议可以用不同措辞表达。"
"""

from src.utils.logger import logger


class AnswerQualityEvaluator:
    """答案质量评测器"""

    def __init__(self):
        self._bert_model = "bert-base-chinese"

    def evaluate_batch(
        self,
        test_pairs: list[dict],
        generate_fn,
    ) -> dict:
        """
        批量评测答案质量。

        Args:
            test_pairs: [{question, answer, id, ...}, ...]
            generate_fn: (query: str) -> str (生成答案)

        Returns:
            {bleu: 0.XX, rouge_l: 0.XX, bert_score: 0.XX}
        """
        scores = {"bleu": [], "rouge_l": [], "bert_score": []}

        for i, pair in enumerate(test_pairs):
            query = pair["question"]
            ground_truth = pair["answer"]
            generated = generate_fn(query)

            scores["bleu"].append(self._bleu(generated, ground_truth))
            scores["rouge_l"].append(self._rouge_l(generated, ground_truth))

            if (i + 1) % 20 == 0:
                logger.info(f"答案评测进度: {i + 1}/{len(test_pairs)}")

        # 计算 BERTScore（批量，效率高）
        bert_scores = self._bert_score_batch(test_pairs, generate_fn)
        scores["bert_score"] = bert_scores

        summary = {}
        for metric, values in scores.items():
            if values:
                summary[metric] = round(sum(values) / len(values), 4)

        summary["total_pairs"] = len(test_pairs)
        logger.info(f"答案质量评测完成: {summary}")
        return summary

    # ============ BLEU ============

    @staticmethod
    def _bleu(reference: str, candidate: str) -> float:
        """BLEU-4（简化实现，不依赖 nltk）"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref_tokens = list(reference)
            cand_tokens = list(candidate)
            return sentence_bleu(
                [ref_tokens], cand_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method1,
            )
        except ImportError:
            return AnswerQualityEvaluator._simple_bleu(reference, candidate)

    @staticmethod
    def _simple_bleu(ref: str, cand: str) -> float:
        """降级：字符级 4-gram precision"""
        def ngrams(s, n):
            return {s[i:i + n] for i in range(len(s) - n + 1)}

        if len(cand) < 4:
            return 1.0 if ref == cand else 0.0

        matches = 0
        for n in range(1, 5):
            ref_ng = ngrams(ref, n)
            cand_ng = ngrams(cand, n)
            if ref_ng:
                matches += len(cand_ng & ref_ng) / len(cand_ng)
        return matches / 4

    # ============ ROUGE-L ============

    @staticmethod
    def _rouge_l(reference: str, candidate: str) -> float:
        """ROUGE-L：最长公共子序列"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            scores = scorer.score(reference, candidate)
            return scores["rougeL"].fmeasure
        except ImportError:
            return AnswerQualityEvaluator._simple_lcs(reference, candidate)

    @staticmethod
    def _simple_lcs(ref: str, cand: str) -> float:
        """降级：LCS 除以 ref 长度"""
        m, n = len(ref), len(cand)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == cand[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        recall = lcs / m
        precision = lcs / n if n > 0 else 0
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    # ============ BERTScore ============

    def _bert_score_batch(
        self,
        test_pairs: list[dict],
        generate_fn,
    ) -> list[float]:
        """BERTScore 批量计算"""
        try:
            from bert_score import score as bert_score_fn

            refs = [p["answer"] for p in test_pairs]
            cands = [generate_fn(p["question"]) for p in test_pairs]

            P, R, F1 = bert_score_fn(
                cands, refs,
                model_type=self._bert_model,
                lang="zh",
                verbose=False,
            )
            return [float(f) for f in F1]
        except ImportError:
            logger.warning("bert_score 未安装，BERTScore 指标不可用")
            return []
        except Exception as e:
            logger.error(f"BERTScore 计算失败: {e}")
            return []
