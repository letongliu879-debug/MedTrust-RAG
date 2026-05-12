"""LangGraph State Machine for MedTrust-RAG Dual-Verify Pipeline

3-Round + Convergence Verify 管线：
  START → RETRIEVE → GENERATE → VERIFY → CONV CHECK
                                        ↓
                            CONVERGED? ←→ REGENERATE (loop)
                                        ↓
                                    SYNTHESIZE → END

异常处理作为一等公民节点：
  VERIFY/REGENERATE 异常 → ERROR → END (fallback answer)

关键设计：
  - evidence_chunks 固定（检索一次，不重新检索）
  - MAX_ITERS = 3（硬上限）
  - 收敛条件：prev_flagged >= curr_flagged（标记段落不再增加）
  - 结构化反馈："Segment[N] claims Y, but evidence shows Z"
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, END

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


# ─── 全局常量 ───────────────────────────────────────────────

MAX_ITERS = 3


# ─── Shared State ───────────────────────────────────────────

class PipelineStatus(str, Enum):
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    ERROR = "error"
    RETRIEVE_FAILED = "retrieve_failed"


@dataclass
class VerificationState:
    """LangGraph 共享状态"""
    query: str = ""
    department: str | None = None
    model_key: str | None = None
    evidence_chunks: list[dict] = field(default_factory=list)
    iteration: int = 0
    draft_answer: str | None = None
    safety_report: SafetyReport | None = None
    prev_flagged_count: int = 0
    feedback_history: list[str] = field(default_factory=list)
    final_answer: str | None = None
    confidence: float = 0.0
    status: PipelineStatus = PipelineStatus.RUNNING
    error_message: str | None = None


# ─── 辅助 ───────────────────────────────────────────────────

def _build_structured_feedback(safety: SafetyReport) -> str:
    """将 safety_report 转换为结构化反馈字符串"""
    lines = []
    for i, seg in enumerate(safety.flagged_segments, 1):
        lines.append(f"  Segment[{i}]: {seg}")
    if safety.suggestions:
        lines.append("  Suggestions:")
        for s in safety.suggestions:
            lines.append(f"    - {s}")
    if safety.contradictions:
        lines.append("  Contradictions:")
        for c in safety.contradictions:
            lines.append(f"    - {c}")
    return "\n".join(lines) if lines else "No specific issues flagged."


# ─── LangGraph Async Nodes ─────────────────────────────────

async def _retrieve_node(state: VerificationState) -> VerificationState:
    """1 RETRIEVE：BM25 + Vector 混合检索（固定不重取）"""
    logger.info(f"[LangGraph] RETRIEVE start, query='{state.query[:40]}...'")

    try:
        agent = RetrieverAgent()
        result = await agent.run(
            query=state.query,
            department=state.department,
        )
        chunks = result.citations if result.citations else []

        if not chunks:
            state.status = PipelineStatus.RETRIEVE_FAILED
            logger.warning("[LangGraph] RETRIEVE: no chunks found")
            return state

        state.evidence_chunks = chunks
        state.iteration = 0
        state.prev_flagged_count = 0
        logger.info(f"[LangGraph] RETRIEVE done: {len(chunks)} chunks")
        return state

    except Exception as e:
        logger.error(f"[LangGraph] RETRIEVE exception: {e}")
        state.status = PipelineStatus.ERROR
        state.error_message = f"检索失败: {e}"
        return state


async def _generate_node(state: VerificationState) -> VerificationState:
    """2 GENERATE：ResponderAgent 生成初稿答案"""
    logger.info(f"[LangGraph] GENERATE start, iter={state.iteration}")

    try:
        agent = ResponderAgent()
        result = await agent.run(
            query=state.query,
            evidence_chunks=state.evidence_chunks,
            model_key=state.model_key,
        )
        state.draft_answer = result.answer
        state.confidence = result.confidence
        logger.info(f"[LangGraph] GENERATE done: confidence={result.confidence:.2f}")
        return state

    except Exception as e:
        logger.error(f"[LangGraph] GENERATE exception: {e}")
        state.status = PipelineStatus.ERROR
        state.error_message = f"生成失败: {e}"
        return state


async def _verify_node(state: VerificationState) -> VerificationState:
    """3 VERIFY：SafetyCheckerAgent 验证答案安全性"""
    logger.info(f"[LangGraph] VERIFY start, iter={state.iteration}")

    try:
        agent = SafetyCheckerAgent()
        result = await agent.run(
            query=state.query,
            answer=state.draft_answer or "",
            evidence_chunks=state.evidence_chunks,
            model_key=state.model_key,
        )
        state.safety_report = result.safety
        logger.info(
            f"[LangGraph] VERIFY done: risk={result.safety.risk_level}, "
            f"flagged={len(result.safety.flagged_segments)}"
        )
        return state

    except Exception as e:
        logger.error(f"[LangGraph] VERIFY exception: {e}")
        state.status = PipelineStatus.ERROR
        state.error_message = f"安全校验失败: {e}"
        return state


async def _conv_check_node(state: VerificationState) -> VerificationState:
    """CONV CHECK：判断是否收敛或需要重生成"""
    safety = state.safety_report
    if safety is None:
        state.status = PipelineStatus.CONVERGED
        return state

    curr_flagged = len(safety.flagged_segments)

    converged = (
        curr_flagged <= state.prev_flagged_count or
        state.iteration >= MAX_ITERS
    )

    if converged:
        if safety.risk_level == "unsafe" and state.iteration < MAX_ITERS:
            state.status = PipelineStatus.RUNNING
        else:
            state.status = PipelineStatus.CONVERGED
            logger.info(
                f"[LangGraph] CONV CHECK: converged=True, "
                f"prev_flagged={state.prev_flagged_count}, "
                f"curr_flagged={curr_flagged}, iter={state.iteration}"
            )
    else:
        state.status = PipelineStatus.RUNNING
        logger.info(
            f"[LangGraph] CONV CHECK: converged=False, "
            f"prev_flagged={state.prev_flagged_count}, "
            f"curr_flagged={curr_flagged}"
        )

    return state


async def _regenerate_node(state: VerificationState) -> VerificationState:
    """4 REGENERATE：基于结构化反馈重生成答案（不重取证据）"""
    logger.info(f"[LangGraph] REGENERATE start, iter={state.iteration}")

    safety = state.safety_report
    feedback = _build_structured_feedback(safety) if safety else "No issues flagged."

    try:
        # 带结构化反馈的 prompt
        from src.llm.model_factory import ModelFactory
        from src.llm.chains import parse_json_from_text
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        evidence_text = ResponderAgent._format_evidence(state.evidence_chunks)
        feedback_prompt = (
            f"请基于以下安全审核反馈修正答案。\n\n"
            f"【原始问题】{state.query}\n\n"
            f"【安全审核反馈】\n{feedback}\n\n"
            f"【参考要求】\n"
            f"- 仅基于参考资料回答，不要编造信息\n"
            f"- 每条关键医学陈述需注明引用编号（如 [参考1]）\n"
            f"- 参考资料不足时明确说明\"根据目前资料，无法完全回答此问题\"\n"
            f"- 使用通俗中文\n"
            f"- 必须修正被标记的问题段落\n\n"
            f"【输出格式】\n"
            f'{{"answer": "详细回答...", "citations": [{{"ref_id": 1, "excerpt": "引用的关键信息..."}}], "confidence": 0.0-1.0}}\n\n'
            f"【参考资料】\n{evidence_text}"
        )

        llm = ModelFactory.create_chat_model(state.model_key)
        prompt = ChatPromptTemplate.from_template("{feedback}")
        chain = prompt | llm | StrOutputParser()
        result = await chain.ainvoke({"feedback": feedback_prompt})
        parsed = parse_json_from_text(result)

        prev_flagged = len(safety.flagged_segments) if safety else 0
        state.prev_flagged_count = prev_flagged
        state.iteration += 1
        state.draft_answer = parsed.get("answer", result)
        state.confidence = float(parsed.get("confidence", 0.5))
        state.feedback_history.append(feedback)

        logger.info(
            f"[LangGraph] REGENERATE done: iter={state.iteration}, "
            f"confidence={state.confidence:.2f}"
        )
        return state

    except Exception as e:
        logger.error(f"[LangGraph] REGENERATE exception: {e}")
        state.status = PipelineStatus.ERROR
        state.error_message = f"重生成失败: {e}"
        return state


async def _synthesize_node(state: VerificationState) -> VerificationState:
    """5 SYNTHESIZE：SynthesizerAgent 合成最终答案"""
    logger.info("[LangGraph] SYNTHESIZE start")

    if state.status == PipelineStatus.RETRIEVE_FAILED:
        state.final_answer = (
            "知识库中暂无足够的相关医学资料来回答此问题。"
            "建议您咨询专业医生获取准确信息。"
        )
        state.confidence = 0.0
        logger.info("[LangGraph] SYNTHESIZE: retrieve failed, using fallback")
        return state

    try:
        agent = SynthesizerAgent()
        result = await agent.run(
            query=state.query,
            answer=state.draft_answer or "",
            safety=state.safety_report or SafetyReport(),
            evidence_chunks=state.evidence_chunks,
            confidence=state.confidence,
            model_key=state.model_key,
        )
        state.final_answer = result.answer
        state.confidence = result.confidence
        logger.info(f"[LangGraph] SYNTHESIZE done: confidence={result.confidence:.2f}")
        return state

    except Exception as e:
        logger.error(f"[LangGraph] SYNTHESIZE exception: {e}")
        safety = state.safety_report or SafetyReport()
        state.final_answer = SynthesizerAgent._fallback_answer(
            state.draft_answer or "", safety
        )
        state.status = PipelineStatus.ERROR
        return state


async def _error_node(state: VerificationState) -> VerificationState:
    """ERROR：异常处理节点，生成 fallback 答案"""
    logger.warning(f"[LangGraph] ERROR node: {state.error_message}")

    safety = state.safety_report or SafetyReport()
    state.final_answer = SynthesizerAgent._fallback_answer(
        state.draft_answer or "", safety
    )
    state.status = PipelineStatus.ERROR

    logger.info("[LangGraph] ERROR node: fallback answer set")
    return state


# ─── 边路由 ─────────────────────────────────────────────────

def _route_after_verify(state: VerificationState) -> Literal["conv_check", "error"]:
    """VERIFY 之后"""
    if state.status == PipelineStatus.ERROR:
        return "error"
    return "conv_check"


def _route_after_conv_check(state: VerificationState) -> Literal["regenerate", "synthesize"]:
    """CONV CHECK 之后"""
    if state.status == PipelineStatus.CONVERGED:
        return "synthesize"
    return "regenerate"


def _route_after_regenerate(state: VerificationState) -> Literal["verify", "error"]:
    """REGENERATE 之后"""
    if state.status == PipelineStatus.ERROR:
        return "error"
    return "verify"


# ─── 构建 StateGraph ─────────────────────────────────────────

def _build_graph() -> StateGraph:
    """构建 LangGraph 状态机"""
    builder = StateGraph(VerificationState)

    # 节点（async）
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("generate", _generate_node)
    builder.add_node("verify", _verify_node)
    builder.add_node("conv_check", _conv_check_node)
    builder.add_node("regenerate", _regenerate_node)
    builder.add_node("synthesize", _synthesize_node)
    builder.add_node("error", _error_node)

    # 起点
    builder.set_entry_point("retrieve")

    # 固定边
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "verify")
    builder.add_edge("synthesize", END)
    builder.add_edge("error", END)

    # 条件边：verify → conv_check / error
    builder.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "conv_check": "conv_check",
            "error": "error",
        },
    )

    # 条件边：conv_check → regenerate / synthesize
    builder.add_conditional_edges(
        "conv_check",
        _route_after_conv_check,
        {
            "regenerate": "regenerate",
            "synthesize": "synthesize",
        },
    )

    # 条件边：regenerate → verify / error
    builder.add_conditional_edges(
        "regenerate",
        _route_after_regenerate,
        {
            "verify": "verify",
            "error": "error",
        },
    )

    return builder.compile()


# ─── Pipeline 包装 ──────────────────────────────────────────

class LangGraphMedicalPipeline:
    """LangGraph 驱动的医疗 QA 管线"""

    def __init__(self):
        self._graph = _build_graph()
        setup_langsmith()
        logger.info("[LangGraph] Pipeline initialized")

    async def arun(
        self,
        query: str,
        department: str = None,
        model_key: str = None,
        on_progress=None,
    ):
        """异步入口（在已有事件循环中使用，如 FastAPI）"""
        from src.utils.cache import semantic_cache

        # 语义缓存检查
        cached = semantic_cache.lookup(query)
        if cached:
            logger.info("[LangGraph] Cache hit, skipping pipeline")
            return _make_report_from_cache(query, cached)

        trace_ctx = start_trace(query)
        start_log(query)

        initial_state = VerificationState(
            query=query,
            department=department,
            model_key=model_key,
        )

        # 执行状态机（已在 async 上下文中，直接 await）
        result_dict = await self._graph.ainvoke(initial_state)
        final_state = VerificationState(**result_dict) if isinstance(result_dict, dict) else result_dict

        finish_trace()
        exec_log_path = finish_log() or ""

        # 存入语义缓存
        if final_state.final_answer:
            semantic_cache.store(query, {
                "answer": final_state.final_answer,
                "confidence": final_state.confidence,
                "risk_level": (
                    final_state.safety_report.risk_level
                    if final_state.safety_report
                    else "safe"
                ),
            })

        report = MedQAReport(
            query=query,
            answer=final_state.final_answer or "处理失败",
            confidence=final_state.confidence,
            citations=final_state.evidence_chunks,
            safety=final_state.safety_report or SafetyReport(),
            model_used=model_key or config.get("llm.default_model"),
            trace={
                "pipeline": "langgraph",
                "iteration": final_state.iteration,
                "status": final_state.status.value,
                "exec_log": exec_log_path,
            },
        )

        if on_progress:
            on_progress("pipeline_done", {
                "status": final_state.status.value,
                "iteration": final_state.iteration,
            })

        return report

    def run(
        self,
        query: str,
        department: str = None,
        model_key: str = None,
        on_progress=None,
    ):
        """同步入口（仅用于 CLI 等非 async 环境）"""
        return asyncio.run(self.arun(query, department, model_key, on_progress))


def _make_report_from_cache(query, cached):
    """从缓存构建报告"""
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


@dataclass
class MedQAReport:
    query: str
    answer: str
    confidence: float
    citations: list[dict] = field(default_factory=list)
    safety: SafetyReport = field(default_factory=SafetyReport)
    model_used: str = ""
    trace: dict = field(default_factory=dict)


# ─── 全局实例 ───────────────────────────────────────────────

langgraph_pipeline = LangGraphMedicalPipeline()
