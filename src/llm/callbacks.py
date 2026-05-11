"""LangChain回调处理器 - 详细日志"""

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from src.utils.logger import logger


class DetailedLoggingCallbackHandler(BaseCallbackHandler):
    """详细LLM日志回调 - 记录prompt、response、token用量、耗时"""

    def __init__(self, reviewer_name: str = "unknown"):
        self.reviewer_name = reviewer_name
        self._start_time = None
        self._completion_tokens = 0

    def on_llm_start(
        self, serialized: dict, prompts: list[str], *, run_id: UUID,
        parent_run_id: UUID | None = None, tags: list[str] | None = None,
        metadata: dict | None = None, **kwargs: Any,
    ) -> None:
        self._start_time = time.time()
        self._completion_tokens = 0
        logger.debug(
            f"[{self.reviewer_name}] LLM请求开始 | "
            f"prompts数量={len(prompts)} | "
            f"prompt长度={sum(len(p) for p in prompts)} | "
            f"tags={tags}"
        )
        for i, prompt in enumerate(prompts):
            truncated = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
            logger.debug(f"[{self.reviewer_name}] Prompt[{i}]: {truncated}")

    def on_llm_new_token(self, token: str, *, run_id: UUID,
                         parent_run_id: UUID | None = None,
                         **kwargs: Any) -> None:
        self._completion_tokens += 1

    def on_llm_end(self, response, *, run_id: UUID,
                   parent_run_id: UUID | None = None,
                   **kwargs: Any) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        token_usage = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})

        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', self._completion_tokens)
        total_tokens = token_usage.get('total_tokens', prompt_tokens + completion_tokens)

        logger.debug(
            f"[{self.reviewer_name}] LLM请求完成 | "
            f"耗时={elapsed:.2f}s | "
            f"prompt_tokens={prompt_tokens} | "
            f"completion_tokens={completion_tokens} | "
            f"total_tokens={total_tokens}"
        )
        for generation in response.generations:
            for gen in generation:
                text = gen.text if hasattr(gen, 'text') else str(gen)
                truncated = text[:2000] + "..." if len(text) > 2000 else text
                logger.debug(f"[{self.reviewer_name}] Response: {truncated}")

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                      parent_run_id: UUID | None = None,
                      **kwargs: Any) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.error(
            f"[{self.reviewer_name}] LLM请求失败 | "
            f"耗时={elapsed:.2f}s | "
            f"错误={type(error).__name__}: {error}"
        )
