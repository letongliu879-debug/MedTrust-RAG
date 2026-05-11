"""语义缓存 — embedding 相似度匹配，命中后跳过全流程

使用方式:
    from src.utils.cache import SemanticCache

    cache = SemanticCache()
    cached = cache.lookup(query)
    if cached:
        return cached
    result = pipeline.run(query)
    cache.store(query, result)
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config_loader import config
from src.utils.logger import logger

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_FILE = CACHE_DIR / "semantic_cache.json"
MAX_CACHE_SIZE = 200  # 最多缓存 200 条


class SemanticCache:
    """基于 embedding 相似度的查询缓存"""

    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold
        self._embeddings: list[np.ndarray] = []
        self._entries: list[dict] = []
        self._embeddings_model = None
        self._loaded = False

    def _ensure_model(self):
        if self._embeddings_model is not None:
            return
        from src.rag.vector_store import vector_store
        self._embeddings_model = vector_store.embeddings

    def _ensure_loaded(self):
        """从磁盘加载缓存"""
        if self._loaded:
            return
        self._loaded = True
        if not CACHE_FILE.exists():
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                emb = np.array(entry.pop("_embedding"), dtype=np.float32)
                self._embeddings.append(emb)
                self._entries.append(entry)
            logger.info(f"语义缓存加载: {len(self._entries)} 条")
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")

    def lookup(self, query: str) -> Optional[dict]:
        """查找缓存。返回 cached_result 或 None"""
        self._ensure_model()
        self._ensure_loaded()

        if not self._entries:
            return None

        query_emb = np.array(self._embeddings_model.embed_query(query), dtype=np.float32)

        if len(self._embeddings) == 0:
            return None

        # 批量计算余弦相似度
        stored = np.stack(self._embeddings)
        # 归一化
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        stored_norm = stored / (np.linalg.norm(stored, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(stored_norm, query_norm)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._threshold:
            logger.info(f"语义缓存命中: similarity={best_score:.4f}, query='{query[:30]}...'")
            return self._entries[best_idx]

        return None

    def store(self, query: str, result: dict):
        """存入缓存"""
        self._ensure_model()
        self._ensure_loaded()

        query_emb = np.array(self._embeddings_model.embed_query(query), dtype=np.float32)

        entry = {
            "query": query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0),
            "risk_level": result.get("risk_level", "safe"),
            "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._embeddings.append(query_emb)
        self._entries.append(entry)

        # LRU 淘汰
        if len(self._entries) > MAX_CACHE_SIZE:
            self._embeddings.pop(0)
            self._entries.pop(0)

        # 异步写盘（不阻塞主流程）
        self._save()

    def _save(self):
        """持久化到磁盘"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            data = []
            for i, entry in enumerate(self._entries):
                entry_copy = dict(entry)
                entry_copy["_embedding"] = self._embeddings[i].tolist()
                data.append(entry_copy)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")

    def clear(self):
        self._embeddings = []
        self._entries = []
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        logger.info("语义缓存已清空")


# 全局实例
semantic_cache = SemanticCache(threshold=config.get("cache.similarity_threshold", 0.95))
