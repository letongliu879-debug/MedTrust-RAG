"""医疗知识库索引工具（并行 + 增量）

用法：
    # 增量索引（默认，跳过已存在的记录）
    python scripts/index_kb.py --sample 1000
    python scripts/index_kb.py --all

    # 强制重建（删旧 collection 全量重来）
    python scripts/index_kb.py --sample 1000 --force
    python scripts/index_kb.py --all --force

    # 按科室分别建 collection
    python scripts/index_kb.py --by-department
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 确保项目根在 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import loader as data_loader
from src.rag.vector_store import vector_store
from src.utils.config_loader import config
from src.utils.logger import logger


class MedicalQAIndexer:
    """将 Huatuo-Lite 的 QA 对索引到 ChromaDB（增量 + 并行）"""

    def __init__(self):
        self._batch_size = config.get("rag.index_batch_size", 20)
        self._collection_name = config.get("dataset.collection_name", "med_all")
        # 并行 worker 数：向 Ollama 发 HTTP 请求的并发数
        # 默认 4（I/O 并发），不是 CPU 核数——embedding 计算在 Ollama 服务端
        # 如果 Ollama 设了 OLLAMA_NUM_PARALLEL > 1，可以相应调大
        self._index_workers = config.get("embedding.index_workers", 4)

    # ============ 全量索引 ============

    def index_all(self, n: int = None, force: bool = False) -> int:
        """全量索引（增量或强制重建）"""
        if n:
            records = data_loader.load_sample(n)
        else:
            records = data_loader.load()
        return self._index_records(records, self._collection_name, force=force)

    def index_sample(self, n: int = 500, force: bool = False) -> int:
        """索引抽样数据"""
        records = data_loader.load_sample(n)
        return self._index_records(records, self._collection_name, force=force)

    # ============ 按科室索引 ============

    def index_by_department(self, n_per_dept: int = None, force: bool = False) -> dict[str, int]:
        """按科室分别建 collection"""
        records = data_loader.load()
        dept_groups: dict[str, list] = {}
        for r in records:
            dept = r["department"] or "未分类"
            dept_groups.setdefault(dept, []).append(r)

        results = {}
        for dept, dept_records in dept_groups.items():
            sample = dept_records[:n_per_dept] if n_per_dept else dept_records
            collection_name = f"med_{dept}"
            count = self._index_records(sample, collection_name, force=force)
            results[dept] = count

        return results

    # ============ 核心索引逻辑 ============

    def _index_records(self, records: list[dict], collection_name: str, force: bool = False) -> int:
        """将 QA 对列表索引入 ChromaDB（增量模式，并行 embedding）"""
        if not records:
            logger.warning("无数据，跳过索引")
            return 0

        # ---- force 模式：删旧重建 ----
        if force:
            try:
                vector_store.client.delete_collection(collection_name)
                logger.info(f"[force] 已删除旧 collection: {collection_name}")
            except Exception:
                pass

        # ---- 获取或创建 collection ----
        collection = vector_store.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Huatuo26M-Lite medical QA pairs"},
        )

        # ---- 获取已存在 ID 集合（增量模式） ----
        existing_ids: set[str] = set()
        if not force:
            try:
                existing_ids = set(collection.get()["ids"])
            except Exception:
                pass

        # ---- 构建待索引的 chunk 列表 ----
        chunks = []
        skipped = 0
        for r in records:
            chunk_id = f"med_{r['id']}"
            if chunk_id in existing_ids:
                skipped += 1
                continue
            content = (
                f"问题: {r['question']}\n"
                f"科室: {r['department']}\n"
                f"回答: {r['answer']}"
            )
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_id": r["id"],
                    "department": r["department"],
                    "score": r["score"],
                    "related_diseases": r["related_diseases"],
                    "question": r["question"],
                    "answer": r["answer"],
                },
            })

        if skipped > 0:
            logger.info(f"跳过已存在: {skipped} 条")

        if not chunks:
            logger.info(f"collection {collection_name} 已是最新，无新数据")
            return 0

        # ---- 并行 embedding + upsert ----
        embeddings_model = vector_store.embeddings
        batch_size = self._batch_size
        total_chunks = len(chunks)

        # 将 chunks 按 batch_size 分组
        batches = []
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batches.append(chunks[start:end])

        n_workers = min(self._index_workers, len(batches))
        logger.info(
            f"开始并行索引: {collection_name}, "
            f"新增 {total_chunks} 条, "
            f"{len(batches)} 批次, "
            f"{n_workers} 并行 worker"
        )

        completed = 0

        def embed_and_collect(batch: list[dict]) -> tuple[list[dict], list]:
            """单个 worker：embed 一个 batch，返回 (原 batch, vectors)"""
            texts = [c["content"] for c in batch]
            try:
                vectors = embeddings_model.embed_documents(texts)
            except Exception:
                # batch 失败则逐个
                vectors = [
                    embeddings_model.embed_query(t) for t in texts
                ]
            return batch, vectors

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(embed_and_collect, b): i for i, b in enumerate(batches)}

            for future in as_completed(futures):
                batch, vectors = future.result()
                ids = [c["id"] for c in batch]
                texts = [c["content"] for c in batch]
                metadatas = [c["metadata"] for c in batch]

                collection.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )

                completed += len(batch)
                if completed % (batch_size * 10) == 0 or completed == total_chunks:
                    logger.info(f"索引进度: {completed}/{total_chunks}")

        logger.info(
            f"索引完成: {collection_name}, "
            f"新增 {total_chunks} 条, "
            f"跳过 {skipped} 条（已存在）"
        )
        return total_chunks

    # ============ 便捷方法 ============

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def collection_count(self) -> int:
        """返回已索引的数据量"""
        try:
            col = vector_store.client.get_collection(self._collection_name)
            return col.count()
        except Exception:
            return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗知识库索引工具")
    parser.add_argument(
        "--sample", "-s", type=int, default=None,
        help="索引采样数量",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全量索引（约 178K 条）",
    )
    parser.add_argument(
        "--by-department", action="store_true",
        help="按科室分别建 collection",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="强制重建（删除旧 collection 全量重来）",
    )
    args = parser.parse_args()

    config.load(str(project_root / "config" / "settings.yaml"))
    idx = MedicalQAIndexer()

    if args.by_department:
        results = idx.index_by_department(n_per_dept=args.sample, force=args.force)
        total = sum(results.values())
        for dept, count in sorted(results.items(), key=lambda x: -x[1])[:10]:
            print(f"  {dept}: {count} 条")
        print(f"\n按科室索引完成: 共 {total} 条, {len(results)} 个科室")
    elif args.all:
        count = idx.index_all(force=args.force)
        print(f"全量索引完成: {count} 条")
    elif args.sample:
        count = idx.index_sample(args.sample, force=args.force)
        print(f"采样索引完成: {count} 条")
    else:
        parser.print_help()
