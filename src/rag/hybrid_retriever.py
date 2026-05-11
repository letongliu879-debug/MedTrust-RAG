"""混合检索器：BM25（关键词） + 向量（语义） + RRF 融合

======================== 医疗可信问答场景 ========================

### 为什么不能用纯向量检索？
医疗文本的专业术语精确匹配很重要。比如"糖尿病"和"高血糖"在语义
向量空间里相似度很高，但对应的诊疗指南、检查项目可能不同。纯向量
检索会导致：
1. 术语混淆——相近症状但完全不同的疾病被混淆
2. 专业词丢失——"空腹血糖"、"糖化血红蛋白"等专业术语被稀释

### BM25 解决什么问题？
BM25 是经典的词频-逆文档频率算法，它：
- **精确匹配关键词**——"糖尿病"只匹配"糖尿病"，不会混淆"高血糖"
- **对医学条文ID敏感**——"第108条"、"诊疗指南"这类精确匹配
- **与向量检索互补**——一个看语义，一个看关键词

### RRF（Reciprocal Rank Fusion）为什么比加权求和好？
两种检索结果的分数不在同一量纲：
- 向量检索：余弦相似度，范围 [0, 1]
- BM25：词频加权和，范围 [0, +∞)，与文档长度相关

直接加权求和需要做分数归一化，但归一化依赖样本多样性。
RRF 绕开这个问题：不管原始分数，只看排名。
    RRF_score(d) = Σ 1 / (k + rank_i(d))
其中 k=60（经验值），rank_i 是文档在第 i 个检索器中的排名
"""

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.rag.vector_store import vector_store
from src.utils.config_loader import config
from src.utils.logger import logger
from src.utils.exec_log import get_log

# BM25 磁盘缓存目录
from pathlib import Path as _Path
_BM25_CACHE_DIR = _Path(__file__).parent.parent.parent / "data" / "cache"


class HybridRetriever:
    """
    混合检索器：BM25 + 向量，RRF 融合。
    """

    # RRF 常数。k=60 是学术界通用值，来自 TREC 实验的最佳实践。
    # 为什么是 60？因为排名第 60 的文档 RRF 得分约为 1/(60+60) ≈ 0.008，
    # 排名第 1 的文档得分约为 1/(60+1) ≈ 0.016，差距约 2 倍，合理。
    RRF_K = 60

    def __init__(self):
        self._top_k = config.get("rag.retrieval_top_k", 5)
        # BM25 召回数：向量检索的 top_k 通常不够覆盖，BM25 多召回一些候选
        self._bm25_top_k = config.get("rag.hybrid.bm25_top_k", 20)
        self._vector_top_k = config.get("rag.hybrid.vector_top_k", 10)
        # 向量相似度阈值：cosine similarity 低于此值的 chunk 被过滤
        # 医学领域设为 0.5——低于 0.3 的相似度在 bge-m3 的语义空间里常常是
        # "同为疾病/症状相关"但实际完全不同的疾病（如糖尿病 vs 支气管炎）
        self._vector_sim_threshold = config.get("rag.hybrid.vector_similarity_threshold", 0.5)
        self._bm25_indices: dict[str, "BM25Okapi"] = {}
        self._bm25_corpora: dict[str, list[str]] = {}
        self._bm25_id_maps: dict[str, list[str]] = {}  # BM25 索引位置 → ChromaDB ID

    # ============ BM25 索引管理 ============

    def _ensure_bm25_index(self, collection_name: str):
        """
        确保 BM25 索引已构建 —— 三级加载策略：

        Level 0: 内存缓存（进程内二次查询直接命中）
        Level 1: 磁盘 pickle 缓存（~1-2s 反序列化）
        Level 2: ChromaDB metadata tokens 字段（跳过 jieba，~75s）
        Level 3: 全量 jieba 分词（降级，~196s）
        """
        from src.utils.trace import get_trace
        trace_ctx = get_trace()
        step_cm = trace_ctx.step("1a_bm25_index_build") if trace_ctx else None
        step = None
        try:
            step = step_cm.__enter__() if step_cm else None

            # Level 0: 内存缓存
            if collection_name in self._bm25_indices:
                if step:
                    step.metadata["status"] = "memory_cache_hit"
                return

            if step:
                step.metadata["collection"] = collection_name

            try:
                collection = vector_store.client.get_collection(collection_name)
            except Exception:
                logger.warning(f"collection {collection_name} 不存在，BM25 索引不可用")
                return

            total_count = collection.count()
            if total_count == 0:
                logger.warning(f"collection {collection_name} 为空，BM25 索引不可用")
                return

            if step:
                step.metadata["doc_count"] = total_count

            # Level 1: 磁盘 pickle 缓存
            cache_path = _BM25_CACHE_DIR / f"bm25_{collection_name}.pkl"
            if cache_path.exists():
                try:
                    import pickle
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                    if cached.get("doc_count") == total_count:
                        self._bm25_indices[collection_name] = cached["bm25"]
                        self._bm25_corpora[collection_name] = cached["corpus"]
                        self._bm25_id_maps[collection_name] = cached["ids"]
                        logger.info(
                            f"BM25 索引从磁盘缓存加载: {collection_name}, "
                            f"{total_count} 个文档"
                        )
                        if step:
                            step.metadata["status"] = "pickle_cache_hit"
                        return
                    else:
                        logger.info(f"BM25 缓存版本不一致 ({cached.get('doc_count')} vs {total_count})，重建")
                        cache_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"BM25 pickle 缓存加载失败: {e}，降级重建")
                    cache_path.unlink(missing_ok=True)

            # 从 ChromaDB 拉取所有文档 + metadata
            SQL_VARIABLE_LIMIT = 999
            documents = []
            ids = []
            metadatas = []
            for offset in range(0, total_count, SQL_VARIABLE_LIMIT):
                batch_data = collection.get(
                    include=["documents", "metadatas"],
                    offset=offset,
                    limit=SQL_VARIABLE_LIMIT,
                )
                documents.extend(batch_data.get("documents", []))
                ids.extend(batch_data.get("ids", []))
                metadatas.extend(batch_data.get("metadatas", []))

            if not documents:
                logger.warning(f"collection {collection_name} 无文档")
                return

            # Level 2: 从 metadata 的 tokens 字段构建（跳过 jieba）
            has_tokens = all(m.get("tokens") for m in metadatas)
            if has_tokens:
                logger.info(f"BM25 从预分词 tokens 构建: {collection_name}")
                corpus = [m["tokens"].split() for m in metadatas]
                if step:
                    step.metadata["status"] = "tokens_fast_path"
            else:
                # Level 3: 全量 jieba 分词（降级）
                logger.info(f"BM25 全量 jieba 分词构建: {collection_name}")
                try:
                    import jieba
                except ImportError:
                    logger.warning("jieba 未安装，使用 2-gram 分词（效果会下降）")
                    jieba = None

                def tokenize(text: str) -> list[str]:
                    if jieba:
                        tokens = jieba.lcut(text)
                        return [t.strip() for t in tokens if t.strip()]
                    else:
                        chars = [c for c in text if c.strip()]
                        return [chars[i:i + 2] for i in range(len(chars) - 1)]

                corpus = [tokenize(doc) for doc in documents]
                if step:
                    step.metadata["status"] = "jieba_full"

            from rank_bm25 import BM25Okapi
            bm25_index = BM25Okapi(corpus)
            self._bm25_indices[collection_name] = bm25_index
            self._bm25_corpora[collection_name] = corpus
            self._bm25_id_maps[collection_name] = ids

            # 写入磁盘 pickle 缓存（下次启动 ~1-2s 加载）
            try:
                import pickle
                _BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "bm25": bm25_index,
                        "corpus": corpus,
                        "ids": ids,
                        "doc_count": total_count,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"BM25 索引已缓存到磁盘: {cache_path}")
            except Exception as e:
                logger.warning(f"BM25 pickle 缓存写入失败: {e}")

            logger.info(
                f"BM25 索引构建完成: {collection_name}, "
                f"{len(corpus)} 个文档, "
                f"来源: {'tokens' if has_tokens else 'jieba'}"
            )
        except Exception as e:
            logger.error(f"BM25 索引构建失败: {e}")
            if step:
                step.metadata["status"] = "error"
                step.metadata["error"] = str(e)
        finally:
            if step_cm:
                step_cm.__exit__(None, None, None)

    # ============ 检索 ============

    @traceable(
        run_type="retriever",
        name="混合检索(RRF)",
        metadata={"stage": "hybrid_retrieval"},
    )
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = None,
        department: str = None,
    ) -> list[dict]:
        """
        混合检索：BM25 + 向量 → RRF 融合。

        支持科室过滤：传 department 时只检索该科室的 chunk（metadata 过滤）。

        Returns:
            [{"text": ..., "metadata": ..., "score": RRF分数, "sources": [...]}, ...]
        """
        k = top_k or self._top_k

        # 1. BM25 检索
        bm25_results = self._bm25_search(collection_name, query, self._bm25_top_k, department)

        # 2. 向量检索
        vector_results = self._vector_search(collection_name, query, self._vector_top_k, department)

        # 3. RRF 融合
        fused = self._rrf_fuse(bm25_results, vector_results, k)

        logger.info(
            f"混合检索: query='{query[:30]}...', "
            f"BM25={len(bm25_results)}条, "
            f"向量={len(vector_results)}条, "
            f"融合后={len(fused)}条"
        )

        # LangSmith trace: 附上粗排详情
        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({
                "bm25_total": len(bm25_results),
                "bm25_top5": [
                    {"rank": r["bm25_rank"], "bm25_score": round(r.get("bm25_score", 0), 4),
                     "text": r["text"][:80]}
                    for r in bm25_results[:5]
                ],
                "vector_total": len(vector_results),
                "vector_top5": [
                    {"similarity": r.get("similarity", 0), "text": r["text"][:80]}
                    for r in vector_results[:5]
                ],
                "fused_total": len(fused),
                "vector_threshold": self._vector_sim_threshold,
            })
            rt.extra = {
                "bm25_top5": [
                    {"rank": r["bm25_rank"], "bm25_score": round(r.get("bm25_score", 0), 4),
                     "text": r["text"][:80]}
                    for r in bm25_results[:5]
                ],
                "vector_top5": [
                    {"similarity": r.get("similarity", 0), "text": r["text"][:80]}
                    for r in vector_results[:5]
                ],
            }
        return fused

    # ============ BM25 检索 ============

    @traceable(
        run_type="retriever",
        name="BM25粗排",
        metadata={"stage": "coarse_bm25"},
    )
    def _bm25_search(self, collection_name: str, query: str, top_k: int, department: str = None) -> list[dict]:
        """BM25 关键词检索（支持科室过滤）"""
        if collection_name not in self._bm25_indices:
            return []

        bm25 = self._bm25_indices[collection_name]
        ids = self._bm25_id_maps[collection_name]

        # BM25 需要对查询分词，否则中文查询会被当单个长字符串处理
        query_tokens = self._tokenize_query(query)

        try:
            scores = bm25.get_scores(query_tokens)
        except Exception:
            return []

        # 如有科室过滤，多拉一些候选以补偿过滤损失
        fetch_k = top_k * 3 if department else top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:fetch_k]

        # 从 ChromaDB 获取实际文档内容（带科室过滤）
        collection = vector_store.client.get_collection(collection_name)
        chroma_ids = [ids[i] for i in top_indices if i < len(ids)]
        if not chroma_ids:
            return []

        where_clause = {"department": department} if department else None
        fetched = collection.get(
            ids=chroma_ids,
            include=["documents", "metadatas"],
            where=where_clause,
        )

        # fetched 可能因为 where 过滤而不包含所有 chroma_ids
        fetched_ids = fetched.get("ids", [])
        fetched_docs = fetched.get("documents", [])
        fetched_metas = fetched.get("metadatas", [])
        fetched_id_set = set(fetched_ids)
        results = []
        for i, idx in enumerate(top_indices):
            if idx >= len(ids):
                continue
            cid = ids[idx]
            if cid not in fetched_id_set:
                continue  # 被科室过滤掉了
            try:
                chroma_idx = fetched_ids.index(cid)
            except ValueError:
                continue
            doc_text = fetched_docs[chroma_idx] if chroma_idx < len(fetched_docs) else ""
            metadata = fetched_metas[chroma_idx] if chroma_idx < len(fetched_metas) else {}
            results.append({
                "text": doc_text,
                "metadata": metadata,
                "bm25_score": float(scores[idx]),
                "bm25_rank": len(results) + 1,
                "sources": ["bm25"],
            })
            if len(results) >= top_k:
                break

        # LangSmith trace: BM25 检索详情
        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({"bm25_returned": len(results)})
            rt.extra = {
                "bm25_top_scores": [
                    {"rank": r["bm25_rank"], "score": round(r["bm25_score"], 4),
                     "text_head": r["text"][:60]}
                    for r in results[:5]
                ],
            }

        # exec_log
        elog = get_log()
        if elog:
            elog.step("bm25_search",
                input={"query_tokens": query_tokens, "top_k": top_k},
                output=[{"rank": r["bm25_rank"], "score": round(r["bm25_score"], 4),
                         "text": r["text"][:120]}
                        for r in results[:15]],
                metadata={"total_returned": len(results)},
            )

        return results

    # ============ 向量检索 ============

    @traceable(
        run_type="retriever",
        name="向量粗排",
        metadata={"stage": "coarse_vector"},
    )
    def _vector_search(self, collection_name: str, query: str, top_k: int, department: str = None) -> list[dict]:
        """向量语义检索（带相似度阈值过滤 + 科室过滤）"""
        from src.utils.trace import get_trace
        trace_ctx = get_trace()
        embed_step_cm = trace_ctx.step("1b0_embed") if trace_ctx else None
        embed_step = None
        try:
            collection = vector_store.client.get_collection(collection_name)
            if collection.count() == 0:
                return []

            try:
                embed_step = embed_step_cm.__enter__() if embed_step_cm else None
                query_embedding = vector_store.embeddings.embed_query(query)
                if embed_step:
                    embed_step.metadata["embedding_dim"] = len(query_embedding)
            finally:
                if embed_step_cm:
                    embed_step_cm.__exit__(None, None, None)

            where_clause = {"department": department} if department else None
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
                where=where_clause,
            )

            retrieved = []
            filtered_count = 0
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                dists = results.get("distances", [[]])[0] if results.get("distances") else []
                metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                for i, doc in enumerate(docs):
                    distance = dists[i] if i < len(dists) else 0
                    # ChromaDB 使用 cosine distance: 1 - cos_sim
                    # 转换为相似度再比较阈值
                    similarity = max(0.0, 1.0 - float(distance))
                    if similarity < self._vector_sim_threshold:
                        filtered_count += 1
                        continue
                    retrieved.append({
                        "text": doc,
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": distance,
                        "similarity": round(similarity, 4),
                        "sources": ["vector"],
                    })

            if filtered_count > 0:
                doc_count = len(results.get("documents", [[]])[0]) if results else 0
                logger.info(
                    f"向量检索阈值过滤: {filtered_count}/{doc_count} "
                    f"个低相似度 chunk 被丢弃 (阈值={self._vector_sim_threshold})"
                )

            # LangSmith trace: 向量检索详情
            rt = get_current_run_tree()
            if rt:
                rt.add_metadata({
                    "vec_returned": len(retrieved),
                    "vec_filtered": filtered_count,
                    "vec_threshold": self._vector_sim_threshold,
                })
                rt.extra = {
                    "vec_top_similarities": [
                        {"rank": i + 1, "similarity": r.get("similarity", 0),
                         "text_head": r["text"][:60]}
                        for i, r in enumerate(retrieved[:5])
                    ],
                }

            # exec_log
            elog = get_log()
            if elog:
                elog.step("vector_search",
                    input={"query": query[:100], "top_k": top_k, "threshold": self._vector_sim_threshold},
                    output=[{"similarity": r.get("similarity", 0), "text": r["text"][:120]}
                            for r in retrieved[:10]],
                    metadata={"returned": len(retrieved), "filtered": filtered_count},
                )

            return retrieved
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    # ============ RRF 融合 ============

    @traceable(
        run_type="retriever",
        name="RRF融合",
        metadata={"stage": "fusion"},
    )
    def _rrf_fuse(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        RRF（Reciprocal Rank Fusion）融合两路检索结果。

        算法：
            RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))

        只有一路检索返回的文档，另一路贡献为 0。

        面试重点：
        - 为什么不直接加权求和？因为 BM25 分数和余弦距离量纲不同
        - 为什么 k=60？学术界的经验值，让排名差异平滑，不会过度奖励排名第 1
        """
        # 用文档文本的 MD5 前 100 字符做去重 key
        def make_key(text: str) -> str:
            import hashlib
            return hashlib.md5(text[:100].encode()).hexdigest()

        # 建立文档索引：key → 合并后的结果
        doc_map = {}

        # BM25 结果
        for rank, result in enumerate(bm25_results, 1):
            key = make_key(result["text"])
            doc_map[key] = {
                "text": result["text"],
                "metadata": result.get("metadata", {}),
                "sources": ["bm25"],
                "_rrf": 1.0 / (self.RRF_K + rank),
            }

        # 向量结果
        for rank, result in enumerate(vector_results, 1):
            key = make_key(result["text"])
            rrf_score = 1.0 / (self.RRF_K + rank)
            if key in doc_map:
                doc_map[key]["_rrf"] += rrf_score
                doc_map[key]["sources"].append("vector")
            else:
                doc_map[key] = {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "sources": ["vector"],
                    "_rrf": rrf_score,
                }

        # 按 RRF 总分排序
        sorted_docs = sorted(doc_map.values(), key=lambda d: d["_rrf"], reverse=True)

        # 输出格式
        result = []
        for doc in sorted_docs[:top_k]:
            result.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(doc["_rrf"], 6),
                "sources": doc["sources"],
            })

        # LangSmith trace: RRF 融合详情
        rt = get_current_run_tree()
        if rt:
            bm25_only = sum(1 for d in sorted_docs if d["sources"] == ["bm25"])
            vec_only = sum(1 for d in sorted_docs if d["sources"] == ["vector"])
            both = sum(1 for d in sorted_docs if "bm25" in d["sources"] and "vector" in d["sources"])
            rt.add_metadata({
                "bm25_input": len(bm25_results),
                "vector_input": len(vector_results),
                "fused_output": len(result),
                "bm25_only": bm25_only,
                "vector_only": vec_only,
                "both": both,
            })
            rt.extra = {
                "rrf_top_scores": [
                    {"rank": i + 1, "rrf_score": r["score"], "sources": r["sources"],
                     "text_head": r["text"][:60]}
                    for i, r in enumerate(result[:5])
                ],
            }

        # exec_log
        elog = get_log()
        if elog:
            elog.step("rrf_fusion",
                input={"bm25": len(bm25_results), "vector": len(vector_results)},
                output=[{"rrf_score": r["score"], "sources": r["sources"],
                         "text": r["text"][:120]}
                        for r in result],
                metadata={"bm25_only": sum(1 for d in sorted_docs if d["sources"] == ["bm25"]),
                          "vector_only": sum(1 for d in sorted_docs if d["sources"] == ["vector"]),
                          "both": sum(1 for d in sorted_docs if len(d["sources"]) > 1)},
            )

        return result

    # ============ 查询处理 ============

    def _tokenize_query(self, query: str) -> list[str]:
        """对查询分词（中文必须分词，否则 BM25 会把整段当单个词）"""
        try:
            import jieba
            return [t.strip() for t in jieba.lcut(query) if t.strip()]
        except ImportError:
            # 降级：2-gram
            chars = [c for c in query if c.strip()]
            return [chars[i:i + 2] for i in range(len(chars) - 1)] or chars


# 全局实例
hybrid_retriever = HybridRetriever()
