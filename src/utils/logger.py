"""日志工具"""

import logging
import sys
from pathlib import Path

from src.utils.config_loader import config


def setup_logger(name: str = "contract_review") -> logging.Logger:
    """初始化日志器"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"), logging.INFO)
    fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file")

    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_llm_logger() -> logging.Logger:
    """获取LLM详细日志器（独立日志文件，避免淹没主日志）"""
    llm_logger = logging.getLogger("contract_review.llm")

    if llm_logger.handlers:
        return llm_logger

    log_config = config.get("logging", {})
    log_file = log_config.get("file")

    llm_logger.setLevel(logging.DEBUG)

    # LLM详细日志写入独立文件
    if log_file:
        llm_log_path = Path(log_file).parent / "llm_detailed.log"
        llm_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(llm_log_path), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        llm_logger.addHandler(file_handler)

    return llm_logger


# 全局日志器
logger = setup_logger()
