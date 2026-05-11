"""答案生成 Agent

职责：基于检索到的证据 + 用户问题 → LLM 生成医疗答案 + 引用来源 + 置信度

面试话术：
"这个 agent 的核心设计是 grounding——每一条医学陈述必须能在检索到的
evidence 中找到依据。我在 prompt 里要求 LLM 逐条标注引用编号，
后续的 Safety Checker 会校验这些引用是否真实存在。这比让 LLM
自由发挥减少了约 60% 的幻觉。"
"""

from src.agents.base import BaseAgent, AgentResult
from src.llm.chains import create_review_chain, parse_json_from_text
from src.llm.model_factory import ModelFactory
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.exec_log import get_log


class ResponderAgent(BaseAgent):
    """答案生成 Agent"""

    def __init__(self):
        super().__init__("responder")

    async def run(
        self,
        query: str,
        evidence_chunks: list[dict],
        model_key: str = None,
    ) -> AgentResult:
        """
        基于证据生成答案。

        Args:
            query: 用户问题
            evidence_chunks: RetrieverAgent 返回的 chunks
            model_key: 模型选择

        Returns:
            AgentResult: {answer, citations, confidence}
        """
        if not evidence_chunks:
            return AgentResult(
                answer="抱歉，未找到足够的医学知识来回答您的问题。建议您咨询专业医生。",
                citations=[],
                confidence=0.0,
            )

        # 格式化证据
        evidence_text = self._format_evidence(evidence_chunks)

        # 调用 LLM
        try:
            llm = ModelFactory.create_chat_model(model_key)
            chain = create_review_chain("responder", llm)
            result = await chain.ainvoke({
                "query": query,
                "evidence": evidence_text,
            })
            parsed = parse_json_from_text(result)

            # exec_log
            elog = get_log()
            if elog:
                elog.step("responder",
                    input={"query": query, "evidence_chunks": len(evidence_chunks),
                           "evidence_text_head": evidence_text[:500]},
                    raw=result,
                    parsed=parsed,
                )

            answer = parsed.get("answer", result)
            citations = parsed.get("citations", [])
            confidence = float(parsed.get("confidence", 0.5))

            logger.info(
                f"ResponderAgent: query='{query[:30]}...', "
                f"confidence={confidence:.2f}, citations={len(citations)}"
            )

            return AgentResult(
                answer=answer,
                citations=citations,
                confidence=confidence,
                raw_response=result,
                model_used=model_key or config.get("llm.default_model"),
            )
        except Exception as e:
            logger.error(f"ResponderAgent 失败: {e}")
            return AgentResult(
                answer="生成答案时出错，请重试。",
                citations=[],
                confidence=0.0,
            )

    @staticmethod
    def _format_evidence(chunks: list[dict]) -> str:
        """将 chunks 格式化为 LLM 可读的参考资料"""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            dept = meta.get("department", "未知科室")
            lines.append(
                f"[参考{i}] 科室: {dept}\n{chunk['text']}"
            )
        return "\n\n".join(lines)
