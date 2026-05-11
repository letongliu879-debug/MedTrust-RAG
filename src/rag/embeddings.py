"""Embedding初始化"""

from langchain_ollama import OllamaEmbeddings

from src.utils.config_loader import config
from src.utils.logger import logger


def create_embeddings() -> OllamaEmbeddings:
    """创建Embedding模型实例（Ollama本地）"""
    emb_cfg = config.get("embedding", {})
    embeddings = OllamaEmbeddings(
        model=emb_cfg.get("model_name", "bge-m3"),
        base_url=emb_cfg.get("base_url", "http://localhost:11434"),
    )
    logger.info(f"创建Embedding模型: {emb_cfg.get('model_name')} (Ollama)")
    return embeddings
