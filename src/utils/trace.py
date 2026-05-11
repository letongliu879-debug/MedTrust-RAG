"""本地结构化 Trace — 替代 LangSmith 的核心功能

不依赖 LangSmith，自动记录每一步的耗时和元数据。
每次 query 结束输出 JSON trace 到 data/traces/，同时打印控制台耗时摘要。

使用方式:
    from src.utils.trace import TraceCollector

    trace = TraceCollector()
    with trace.step("retrieve", metadata={"sub_queries": 3}):
        ...
    with trace.step("rerank", metadata={"candidates": 15}):
        ...

    trace.summary()       # 打印控制台摘要
    trace.save("query_ts") # 写 JSON 文件
"""

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logger import logger

TRACE_DIR = Path(__file__).parent.parent.parent / "data" / "traces"


@dataclass
class StepRecord:
    name: str
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


class TraceCollector:
    """收集单次 pipeline 运行的所有步骤耗时"""

    def __init__(self, query: str = ""):
        self.query = query
        self.query_hash = hashlib.md5(query.encode()).hexdigest()[:8] if query else "unknown"
        self.steps: list[StepRecord] = []
        self._start_ts = time.perf_counter()
        self._current_step: Optional[StepRecord] = None
        self._step_start: float = 0.0

    @contextmanager
    def step(self, name: str, metadata: dict = None, extra: dict = None):
        """上下文管理器：计录一个步骤的耗时"""
        step = StepRecord(name=name, metadata=metadata or {}, extra=extra or {})
        self.steps.append(step)
        t0 = time.perf_counter()
        try:
            yield step
        finally:
            step.duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start_ts) * 1000, 1)

    def summary(self) -> str:
        """控制台摘要表格"""
        total = self.total_ms()
        lines = [
            "",
            "=" * 62,
            f"  Trace: {self.query[:40]}{'...' if len(self.query) > 40 else ''}",
            f"  Total: {total:.0f}ms ({total/1000:.1f}s)",
            "-" * 62,
        ]
        for s in self.steps:
            pct = (s.duration_ms / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            meta_str = ""
            if s.metadata:
                meta_items = [f"{k}={v}" for k, v in s.metadata.items()
                              if k not in ("stage",)]
                if meta_items:
                    meta_str = f"  [{', '.join(meta_items)}]"
            lines.append(f"  {s.name:<20} {s.duration_ms:>7.0f}ms ({pct:>4.1f}%) {bar}{meta_str}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "total_ms": self.total_ms(),
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "metadata": s.metadata,
                    "extra": s.extra,
                }
                for s in self.steps
            ],
        }

    def save(self, label: str = "") -> str:
        """写入 JSON trace 文件"""
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{self.query_hash}_{label}.json" if label else f"{ts}_{self.query_hash}.json"
        filepath = TRACE_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Trace saved: {filepath}")
        return str(filepath)

    def print_and_save(self, label: str = "") -> str:
        """打印摘要 + 保存文件"""
        logger.info(self.summary())
        return self.save(label)


# 全局当前 trace（线程不安全，但 pipeline 是单请求串行，够用）
_current_trace: Optional[TraceCollector] = None


def start_trace(query: str) -> TraceCollector:
    """开始一次 trace"""
    global _current_trace
    _current_trace = TraceCollector(query=query)
    return _current_trace


def get_trace() -> Optional[TraceCollector]:
    """获取当前 trace"""
    return _current_trace


def finish_trace(label: str = "") -> Optional[str]:
    """结束并保存当前 trace"""
    global _current_trace
    if _current_trace is None:
        return None
    result = _current_trace.print_and_save(label)
    _current_trace = None
    return result
