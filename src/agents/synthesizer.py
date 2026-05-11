"""共识合成 Agent

职责：综合 safety report，精炼最终答案，加免责声明

面试话术：
"Synthesizer 不是简单拼接，而是按风险等级做差异化处理：
- safe：润色语言，确保引用准确
- caution：修正被标记的段落，加适当警示
- unsafe：重写问题段落，降低置信度，强制加就医建议

这体现了 defensive design——系统的输出不会因为某个 agent 出错
就直接暴露给用户。"
"""

from src.agents.base import BaseAgent, AgentResult, SafetyReport
from src.llm.chains import create_review_chain, parse_json_from_text
from src.llm.model_factory import ModelFactory
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.exec_log import get_log


class SynthesizerAgent(BaseAgent):
    """共识合成 Agent"""

    def __init__(self):
        super().__init__("synthesizer")

    async def run(
        self,
        query: str,
        answer: str,
        safety: SafetyReport,
        evidence_chunks: list[dict],
        confidence: float = 0.5,
        model_key: str = None,
    ) -> AgentResult:
        """综合所有信息，输出最终答案"""
        try:
            llm = ModelFactory.create_chat_model(model_key)
            chain = create_review_chain("synthesizer", llm)

            prompt_input = {
                "query": query,
                "draft_answer": answer,
                "risk_level": safety.risk_level,
                "flagged": self._format_list(safety.flagged_segments),
                "suggestions": self._format_list(safety.suggestions),
                "contradictions": self._format_list(safety.contradictions),
                "evidence_count": str(len(evidence_chunks)),
            }

            result = await chain.ainvoke(prompt_input)
            parsed = parse_json_from_text(result)

            # exec_log
            elog = get_log()
            if elog:
                elog.step("synthesizer",
                    input=prompt_input,
                    raw=result,
                    parsed=parsed,
                )

            final_answer = parsed.get("final_answer", answer)
            final_confidence = float(parsed.get("confidence", confidence))

            # 降置信度：如果安全问题多
            if safety.risk_level == "unsafe":
                final_confidence = min(final_confidence, 0.3)
            elif safety.risk_level == "caution":
                final_confidence = min(final_confidence, 0.7)

            # 确保有免责声明
            if "不构成医疗建议" not in final_answer:
                final_answer += (
                    "\n\n[免责声明] 本回答仅供参考，不构成医疗建议。"
                    "如有健康问题，请及时咨询专业医生。"
                )

            logger.info(
                f"Synthesizer: risk={safety.risk_level}, "
                f"confidence={final_confidence:.2f}"
            )

            return AgentResult(
                answer=final_answer,
                citations=parsed.get("citations", []),
                confidence=final_confidence,
                safety=safety,
                raw_response=result,
                model_used=model_key or config.get("llm.default_model"),
            )
        except Exception as e:
            logger.error(f"Synthesizer 失败: {e}")
            return AgentResult(
                answer=self._fallback_answer(answer, safety),
                confidence=min(confidence, 0.3),
                safety=safety,
            )

    # ============ 辅助 ============

    @staticmethod
    def _format_list(items: list[str]) -> str:
        if not items:
            return "无"
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _fallback_answer(answer: str, safety: SafetyReport) -> str:
        disclaimer = (
            "\n\n[免责声明] 本回答仅供参考，不构成医疗建议。"
            "如有健康问题，请及时咨询专业医生。"
        )
        if safety.risk_level == "unsafe":
            return (
                "经安全校验，原始回答存在潜在风险，已被拦截。"
                "建议您咨询专业医生获取准确信息。"
            )
        return answer + disclaimer
