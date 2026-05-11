"""医疗 QA 主流程编排

串联 4 Agent：Retriever → Responder → Safety Checker → Synthesizer

面试话术：
"管线是串行的——安全校验强依赖答案先生成，每个 agent
内部都是异步的，retriever 的多路子查询也是并行检索的，
也是并行检索的，整体延迟控制在 5-8 秒。"
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field

from src.agents.retriever_agent import RetrieverAgent
from src.agents.responder_agent import ResponderAgent
from src.agents.safety_checker import SafetyCheckerAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.base import AgentResult, SafetyReport
from src.llm.langsmith_setup import setup_langsmith
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.trace import start_trace, finish_trace
from src.utils.exec_log import start_log, finish_log
from src.utils.cache import semantic_cache


def _run_async(coro):
    """安全地从同步上下文运行异步协程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


@dataclass
class MedQAReport:
    """医疗 QA 最终报告"""

    query: str
    answer: str
    confidence: float
    citations: list[dict] = field(default_factory=list)
    safety: SafetyReport = field(default_factory=SafetyReport)
    model_used: str = ""
    trace: dict = field(default_factory=dict)


class MedicalPipeline:
    """医疗 QA 主流程"""

    def __init__(self):
        self._retriever = RetrieverAgent()
        self._responder = ResponderAgent()
        self._safety = SafetyCheckerAgent()
        self._synthesizer = SynthesizerAgent()
        setup_langsmith()

    def run(
        self,
        query: str,
        department: str = None,
        model_key: str = None,
        on_progress=None,
    ) -> MedQAReport:
        """同步入口"""
        return _run_async(self.arun(
            query=query,
            department=department,
            model_key=model_key,
            on_progress=on_progress,
        ))

    async def arun(
        self,
        query: str,
        department: str = None,
        model_key: str = None,
        on_progress=None,
    ) -> MedQAReport:
        """异步主流程：4 Agent 串行（带语义缓存 + 进度回调）

        Args:
            on_progress: 进度回调函数，签名 (step_name: str, metadata: dict) -> None
                         step_name: "retrieve_start" / "retrieve_done" / "responder_start" / ...
                         metadata: 阶段相关信息（chunks数、confidence等）
        """
        def _notify(step: str, meta: dict = None):
            if on_progress:
                on_progress(step, meta or {})

        logger.info(f"MedQA pipeline start: '{query[:50]}...'")

        # 语义缓存检查
        cached = semantic_cache.lookup(query)
        if cached:
            logger.info(f"  [cache] 命中缓存，跳过全流程")
            return MedQAReport(
                query=query,
                answer=cached["answer"],
                confidence=cached["confidence"],
                safety=SafetyReport(
                    is_safe=(cached.get("risk_level", "safe") == "safe"),
                    risk_level=cached.get("risk_level", "safe"),
                ),
                model_used="cached",
                trace={"cached": True, "cached_at": cached.get("cached_at", "")},
            )

        trace_ctx = start_trace(query)
        start_log(query)
        report_trace = {}

        # --- Agent 1: 检索 ---
        _notify("retrieve_start")
        with trace_ctx.step("1_retrieve") as ts:
            evidence = await self._retriever.run(
                query=query,
                department=department,
            )
            ts.metadata["chunks"] = len(evidence.citations)
            ts.metadata["confidence"] = evidence.confidence
        report_trace["retrieved_chunks"] = len(evidence.citations)
        logger.info(f"  [1/4] Retriever: {report_trace['retrieved_chunks']} chunks")
        _notify("retrieve_done", {"chunks": len(evidence.citations), "confidence": evidence.confidence})

        # 检索质量不足 → 提前退出，省去 Agent 2/3/4 的 LLM 调用
        if not evidence.citations:
            logger.info("  [!] 检索失败，知识库未覆盖此问题，跳过后续 Agent")
            finish_trace()
            exec_log_path = finish_log() or ""
            return MedQAReport(
                query=query,
                answer="知识库中暂无足够的相关医学资料来回答此问题。建议您咨询专业医生获取准确信息。",
                confidence=0.0,
                citations=[],
                safety=SafetyReport(is_safe=True, risk_level="caution"),
                model_used=model_key or config.get("llm.default_model"),
                trace={**report_trace, "exec_log": exec_log_path},
            )

        # --- Agent 2: 生成 ---
        _notify("responder_start")
        with trace_ctx.step("2_responder") as ts:
            response = await self._responder.run(
                query=query,
                evidence_chunks=evidence.citations,
                model_key=model_key,
            )
            ts.metadata["confidence"] = response.confidence
        report_trace["responder_confidence"] = response.confidence
        logger.info(f"  [2/4] Responder: confidence={response.confidence:.2f}")
        _notify("responder_done", {"confidence": response.confidence})

        # --- Agent 3: 安全校验 ---
        _notify("safety_start")
        with trace_ctx.step("3_safety_check") as ts:
            safety_result = await self._safety.run(
                query=query,
                answer=response.answer,
                evidence_chunks=evidence.citations,
                model_key=model_key,
            )
            ts.metadata["risk_level"] = safety_result.safety.risk_level
            ts.metadata["flagged"] = len(safety_result.safety.flagged_segments)
        report_trace["risk_level"] = safety_result.safety.risk_level
        logger.info(f"  [3/4] Safety: risk={safety_result.safety.risk_level}")
        _notify("safety_done", {"risk_level": safety_result.safety.risk_level, "flagged": len(safety_result.safety.flagged_segments)})

        # --- Agent 4: 合成 ---
        _notify("synthesize_start")
        with trace_ctx.step("4_synthesize") as ts:
            final = await self._synthesizer.run(
                query=query,
                answer=response.answer,
                safety=safety_result.safety,
                evidence_chunks=evidence.citations,
                confidence=response.confidence,
                model_key=model_key,
            )
            ts.metadata["confidence"] = final.confidence
        report_trace["final_confidence"] = final.confidence
        logger.info(f"  [4/4] Synthesizer: confidence={final.confidence:.2f}")
        _notify("synthesize_done", {"confidence": final.confidence})

        finish_trace()
        exec_log_path = finish_log() or ""

        # 存入语义缓存
        semantic_cache.store(query, {
            "answer": final.answer,
            "confidence": final.confidence,
            "risk_level": safety_result.safety.risk_level,
        })

        return MedQAReport(
            query=query,
            answer=final.answer,
            confidence=final.confidence,
            citations=final.citations or evidence.citations,
            safety=final.safety,
            model_used=model_key or config.get("llm.default_model"),
            trace={**report_trace, "exec_log": exec_log_path},
        )


# 全局实例
pipeline = MedicalPipeline()
