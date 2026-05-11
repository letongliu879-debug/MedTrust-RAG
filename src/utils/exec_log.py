"""本地执行日志 — 每次查询生成一个 JSON，等同 LangSmith trace

用法:
    from src.utils.exec_log import start_log, get_log, finish_log

    start_log("高血压怎么治")
    log = get_log()
    log.step("bm25", input={"query": q}, output={"results": [...]})
    log.step("reranker", input={"docs": 15}, output={"top5": [...]})
    finish_log()  # → debug_logs/20260510_143052_a1b2c3d4.json
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path(__file__).parent.parent.parent / "debug_logs"

_current_log: Optional["ExecLog"] = None


class ExecLog:
    """单次查询的完整执行日志"""

    def __init__(self, query: str):
        self.query = query
        self.query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        self.start_ts = time.time()
        self.steps: list[dict] = []
        self._order = 0

    def step(self, name: str, **kwargs: Any):
        """记录一个步骤。kwargs 键名自由，常见: input, output, parsed, raw, metadata"""
        self._order += 1
        record = {
            "order": self._order,
            "step": name,
            "elapsed_ms": round((time.time() - self.start_ts) * 1000),
        }
        record.update({k: self._truncate(v) for k, v in kwargs.items()})
        self.steps.append(record)

    def save(self) -> str:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{self.query_hash}.json"
        filepath = LOG_DIR / filename

        doc = {
            "query": self.query,
            "total_ms": round((time.time() - self.start_ts) * 1000),
            "steps": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return str(filepath)

    @staticmethod
    def _truncate(obj: Any, max_str: int = 1000, max_list: int = 20) -> Any:
        if isinstance(obj, str):
            return obj if len(obj) <= max_str else obj[:max_str] + f"...[{len(obj)}chars]"
        if isinstance(obj, dict):
            return {k: ExecLog._truncate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) > max_list:
                return [ExecLog._truncate(x) for x in obj[:max_list]] + [
                    f"...[{len(obj) - max_list} more]"
                ]
            return [ExecLog._truncate(x) for x in obj]
        return obj


def start_log(query: str) -> ExecLog:
    global _current_log
    _current_log = ExecLog(query)
    return _current_log


def get_log() -> Optional[ExecLog]:
    return _current_log


def finish_log() -> Optional[str]:
    global _current_log
    if _current_log is None:
        return None
    path = _current_log.save()
    _current_log = None
    return path
