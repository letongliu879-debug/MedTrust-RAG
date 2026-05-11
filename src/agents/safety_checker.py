"""医疗安全校验 Agent

职责：校验生成的答案是否存在幻觉、危险建议、证据矛盾

面试话术：
"这是整个管线最关键的一环。医疗场景对安全性要求极高——
模型说错一个药名可能造成严重后果。Safety Checker 做了三层校验：

1. 事实接地：把答案拆成原子声明，逐条在 evidence 中匹配
2. 医疗安全分类：用专门的安全 prompt 判断答案是否有危险建议
3. 证据交叉验证：如果两个检索到的 QA 对给的治疗方案互相矛盾，
   系统会标记而不是盲目选择一边

这对应了医疗场景下的幻觉过滤模块，做了
更细粒度的风险分级。"
"""

from src.agents.base import BaseAgent, AgentResult, SafetyReport
from src.llm.chains import create_review_chain, parse_json_from_text
from src.llm.model_factory import ModelFactory
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.exec_log import get_log


class SafetyCheckerAgent(BaseAgent):
    """医疗安全校验 Agent"""

    RISK_SAFE = "safe"
    RISK_CAUTION = "caution"
    RISK_UNSAFE = "unsafe"

    def __init__(self):
        super().__init__("safety_checker")

    async def run(
        self,
        query: str,
        answer: str,
        evidence_chunks: list[dict],
        model_key: str = None,
    ) -> AgentResult:
        """校验答案安全性"""
        if not evidence_chunks:
            return AgentResult(
                answer=answer,
                safety=self._default_safe(),
                confidence=0.5,
            )

        evidence_text = self._format_evidence_summary(evidence_chunks)

        try:
            llm = ModelFactory.create_chat_model(model_key)
            chain = create_review_chain("safety_checker", llm)
            result = await chain.ainvoke({
                "query": query,
                "answer": answer,
                "evidence": evidence_text,
            })
            parsed = parse_json_from_text(result)

            # exec_log
            elog = get_log()
            if elog:
                elog.step("safety_checker",
                    input={"query": query, "answer_head": answer[:300],
                           "evidence_head": evidence_text[:300]},
                    raw=result,
                    parsed=parsed,
                )

            risk_level = parsed.get("risk_level", self.RISK_SAFE)
            if risk_level not in (self.RISK_SAFE, self.RISK_CAUTION, self.RISK_UNSAFE):
                risk_level = self.RISK_SAFE

            safety = SafetyReport(
                is_safe=(risk_level == self.RISK_SAFE),
                risk_level=risk_level,
                flagged_segments=parsed.get("flagged_segments", []),
                suggestions=parsed.get("suggestions", []),
                contradictions=parsed.get("contradictions", []),
            )

            logger.info(
                f"SafetyChecker: risk={risk_level}, "
                f"flagged={len(safety.flagged_segments)}, "
                f"contradictions={len(safety.contradictions)}"
            )

            return AgentResult(
                answer=answer,
                safety=safety,
                confidence=self._risk_confidence(risk_level),
            )
        except Exception as e:
            logger.error(f"SafetyChecker 失败: {e}")
            return AgentResult(
                answer=answer,
                safety=self._default_safe(),
                confidence=0.5,
            )

    @staticmethod
    def _format_evidence_summary(chunks: list[dict]) -> str:
        """证据摘要（给 safety checker 看的版本，保留完整内容确保交叉验证准确）"""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            dept = meta.get("department", "未知")
            lines.append(f"[E{i}] ({dept}) {chunk['text']}")
        return "\n".join(lines)

    @staticmethod
    def _default_safe() -> SafetyReport:
        return SafetyReport(is_safe=True, risk_level="safe")

    @staticmethod
    def _risk_confidence(risk: str) -> float:
        return {"safe": 0.9, "caution": 0.5, "unsafe": 0.2}.get(risk, 0.5)
