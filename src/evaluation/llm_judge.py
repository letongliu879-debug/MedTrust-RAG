"""LLM-as-Judge 评测

用 LLM 对生成答案做多维评估：忠实度、安全性、相关性、完整性

面试话术：
"自动化指标（BLEU/ROUGE）能衡量字面相似度，但衡量不了'答案是否靠谱'。
我用 LLM 做裁判，对四个维度打分：

1. 忠实度：答案里的医学声明是否都能在检索证据里找到依据？
2. 安全性：答案有没有危险的建议或遗漏就医提醒？
3. 相关性：答案是否切中用户的问题？
4. 完整性：答案有没有覆盖关键信息？

LLM-judge 的问题是自身有 bias——对自己生成的答案打分偏高。
所以我用不同的模型当裁判（deepseek 生成 → glm 裁判），降低同模型偏好。"
"""

import json

from src.llm.chains import create_review_chain, parse_json_from_text
from src.llm.model_factory import ModelFactory
from src.utils.config_loader import config
from src.utils.logger import logger


class MedicalQAJudge:
    """LLM 裁判评测"""

    DIMENSIONS = ["faithfulness", "safety", "relevance", "completeness"]

    def __init__(self):
        pass

    def evaluate_single(
        self,
        query: str,
        generated: str,
        ground_truth: str,
        evidence: list[str],
        model_key: str = None,
    ) -> dict:
        """评测单条 QA，返回各维度分数 0-1"""
        judge_model = model_key or config.get("evaluation.judge_model", "zhipu")

        try:
            llm = ModelFactory.create_chat_model(judge_model)
            chain = create_review_chain("llm_judge", llm)

            evidence_text = "\n---\n".join(evidence[:5])
            result = chain.invoke({
                "query": query,
                "generated_answer": generated,
                "ground_truth": ground_truth,
                "evidence": evidence_text,
            })
            parsed = parse_json_from_text(result)

            scores = {}
            for dim in self.DIMENSIONS:
                scores[dim] = float(parsed.get(dim, 0.5))
            scores["overall"] = float(parsed.get("overall", sum(scores.values()) / len(self.DIMENSIONS)))

            return scores
        except Exception as e:
            logger.error(f"LLM-judge 评测失败: {e}")
            return {dim: 0.5 for dim in self.DIMENSIONS + ["overall"]}

    def evaluate_batch(
        self,
        test_pairs: list[dict],
        generate_fn,
        n: int = 50,
        model_key: str = None,
    ) -> dict:
        """
        批量评测。

        Args:
            test_pairs: [{question, answer, ...}, ...]
            generate_fn: (query: str) -> dict {answer, evidence: [...]}
            n: 评测条数（LLM 调用贵，限制数量）
            model_key: 裁判模型

        Returns:
            {faithfulness: 0.XX, safety: 0.XX, ...}
        """
        pairs = test_pairs[:n]

        agg = {dim: [] for dim in self.DIMENSIONS + ["overall"]}

        for i, pair in enumerate(pairs):
            gen_result = generate_fn(pair["question"])
            answer = gen_result.get("answer", "") if isinstance(gen_result, dict) else str(gen_result)
            evidence = gen_result.get("evidence", []) if isinstance(gen_result, dict) else []

            scores = self.evaluate_single(
                query=pair["question"],
                generated=answer,
                ground_truth=pair["answer"],
                evidence=evidence,
                model_key=model_key,
            )
            for dim, val in scores.items():
                agg[dim].append(val)

            if (i + 1) % 10 == 0:
                logger.info(f"LLM-judge 进度: {i + 1}/{n}")

        summary = {}
        for dim, values in agg.items():
            if values:
                summary[dim] = round(sum(values) / len(values), 4)

        logger.info(f"LLM-judge 评测完成: {summary}")
        return summary
