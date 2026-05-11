"""LangSmith追踪配置"""

import os
from src.utils.config_loader import config
from src.utils.logger import logger


def setup_langsmith():
    """根据配置初始化LangSmith追踪"""
    langsmith_config = config.get("langsmith", {})
    if not langsmith_config.get("enabled", False):
        logger.debug("LangSmith未启用")
        return

    api_key = langsmith_config.get("api_key", "")
    project = langsmith_config.get("project", "medqa")
    tracing = langsmith_config.get("tracing", True)

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
        logger.info(f"LangSmith已启用: project={project}, tracing={tracing}")
    else:
        logger.warning("LangSmith已启用但未配置API_KEY，追踪功能不可用")
