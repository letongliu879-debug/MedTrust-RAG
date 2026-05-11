"""检索器 - 按合同类型路由到对应的向量库"""

from src.rag.vector_store import vector_store
from src.utils.config_loader import config
from src.utils.logger import logger


class Retriever:
    """知识库检索器，按合同类型路由"""

    def __init__(self):
        self._top_k = config.get("rag.retrieval_top_k", 5)

    def retrieve(self, contract_type: str, query: str, top_k: int = None) -> list[dict]:
        """
        检索相关知识

        Args:
            contract_type: 合同类型
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表，每项包含 text, metadata, distance
        """
        k = top_k or self._top_k

        try:
            collection = vector_store.get_or_create_collection(contract_type)

            if collection.count() == 0:
                logger.warning(f"合同类型{contract_type}的知识库为空")
                return []

            # 生成查询向量
            query_embedding = vector_store.embeddings.embed_query(query)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
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

            logger.info(f"检索{contract_type}知识: query='{query[:30]}...', 返回{len(retrieved)}条")
            return retrieved

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def retrieve_for_review(self, contract_type: str, contract_text: str) -> str:
        """
        为审查准备检索结果，将检索到的知识格式化为文本

        Args:
            contract_type: 合同类型
            contract_text: 合同文本（用于生成查询）

        Returns:
            格式化的法规参考文本
        """
        # 用合同文本的关键部分生成多个查询
        queries = self._generate_queries(contract_text)
        all_results = []

        for query in queries:
            results = self.retrieve(contract_type, query, top_k=3)
            all_results.extend(results)

        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            key = r["text"][:100]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        if not unique_results:
            return "暂无相关法规参考"

        formatted = []
        for i, r in enumerate(unique_results[:10], 1):
            source = r["metadata"].get("file_name", "未知来源")
            formatted.append(f"[{i}] 来源: {source}\n{r['text']}")

        return "\n\n".join(formatted)

    def _generate_queries(self, contract_text: str) -> list[str]:
        """从合同文本生成检索查询"""
        queries = []
        text_len = len(contract_text)

        # 第一部分（控制在100字以内，避免超出embedding模型上下文）
        queries.append(contract_text[:100] if text_len > 100 else contract_text)

        # 中间部分
        if text_len > 500:
            mid = text_len // 2
            queries.append(contract_text[mid:mid + 100])

        # 合同标题/开头
        first_lines = contract_text.split("\n")[:3]
        queries.append("\n".join(first_lines))

        return queries


# 全局实例
retriever = Retriever()
