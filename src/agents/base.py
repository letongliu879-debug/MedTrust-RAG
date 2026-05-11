"""Agent 基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvidenceBundle:
    """检索证据包"""

    chunks: list[dict] = field(default_factory=list)
    query: str = ""
    department_hint: str = ""


@dataclass
class SafetyReport:
    """安全校验报告"""

    is_safe: bool = True
    risk_level: str = "safe"  # safe / caution / unsafe
    flagged_segments: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Agent 输出"""

    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    safety: SafetyReport = field(default_factory=SafetyReport)
    raw_response: str = ""
    model_used: str = ""


class BaseAgent(ABC):
    """所有 Agent 的基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(self, **kwargs) -> AgentResult:
        """执行 agent 逻辑"""
        pass
