"""检索器 - 医疗可信问答知识库检索"""

from src.rag.vector_store import vector_store
from src.utils.config_loader import config
from src.utils.logger import logger


class Retriever:
    """医疗知识库检索器，支持按科室过滤"""

    def __init__(self):
        self._top_k = config.get("rag.retrieval_top_k", 5)

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = None,
        department: str = None,
    ) -> list[dict]:
        """
        检索相关医学知识

        Args:
            query: 查询文本
            collection_name: ChromaDB collection 名称
            top_k: 返回结果数量
            department: 科室名称（可选，用于过滤）

        Returns:
            检索结果列表，每项包含 text, metadata, distance
        """
        k = top_k or self._top_k

        try:
            collection = vector_store.get_or_create_collection(collection_name)

            if collection.count() == 0:
                logger.warning(f"collection {collection_name} 为空")
                return []

            # 生成查询向量
            query_embedding = vector_store.embeddings.embed_query(query)

            # 构建过滤条件
            where_clause = {"department": department} if department else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
                where=where_clause,
            )

            retrieved = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                dists = results.get("distances", [[]])[0] if results.get("distances") else []
                for i, doc in enumerate(docs):
                    retrieved.append({
                        "text": doc,
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": dists[i] if i < len(dists) else 0,
                    })

            logger.info(f"检索 {collection_name}: query='{query[:30]}...', 返回 {len(retrieved)} 条")
            return retrieved

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []


# 全局实例
retriever = Retriever()
