"""Chroma向量存储管理"""

from pathlib import Path

import chromadb

from src.rag.embeddings import create_embeddings
from src.utils.config_loader import config
from src.utils.logger import logger

# 项目根目录（src的上级）
PROJECT_ROOT = Path(__file__).parent.parent.parent


class VectorStore:
    """Chroma向量存储管理器"""

    def __init__(self):
        persist_dir = config.get("rag.chroma_persist_dir", "data/chroma_db")
        # 相对路径基于项目根目录解析，避免工作目录不同导致路径错误
        if not Path(persist_dir).is_absolute():
            persist_dir = str(PROJECT_ROOT / persist_dir)
        self._persist_dir = persist_dir
        self._client = None
        self._embeddings = None
        self._collections = {}

    @property
    def client(self) -> chromadb.PersistentClient:
        """获取Chroma客户端"""
        if self._client is None:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info(f"Chroma客户端初始化: {self._persist_dir}")
        return self._client

    @property
    def embeddings(self):
        """获取Embedding模型"""
        if self._embeddings is None:
            self._embeddings = create_embeddings()
        return self._embeddings

    def get_or_create_collection(self, contract_type: str):
        """获取或创建指定合同类型的collection"""
        collection_name = f"contract_{contract_type}"
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"contract_type": contract_type},
            )
            logger.info(f"获取/创建collection: {collection_name}")
        return self._collections[collection_name]

    def list_collections(self) -> list[str]:
        """列出所有collection"""
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, contract_type: str):
        """删除指定合同类型的collection"""
        collection_name = f"contract_{contract_type}"
        self.client.delete_collection(collection_name)
        self._collections.pop(collection_name, None)
        logger.info(f"删除collection: {collection_name}")


# 全局实例
vector_store = VectorStore()
