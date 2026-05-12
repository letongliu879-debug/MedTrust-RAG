"""MedTrust-RAG 命令行入口

用法:
    python main.py index [--sample 1000]   # 索引 Huatuo-Lite 到 ChromaDB
    python main.py index --all             # 全量索引（约 178K 条）
    python main.py query "高血压怎么治"     # 单次问答
    python main.py eval --sample 100       # 跑评测
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config_loader import config


def cmd_index(args):
    from scripts.index_kb import MedicalQAIndexer
    idx = MedicalQAIndexer()

    if args.all:
        count = idx.index_all(force=args.force)
        print(f"全量索引完成: {count} 条")
    elif args.sample:
        count = idx.index_sample(args.sample, force=args.force)
        print(f"采样索引完成: {count} 条")
    else:
        print("请指定 --sample N 或 --all")
        sys.exit(1)


def cmd_query(args):
    from src.pipeline.langgraph_pipeline import langgraph_pipeline as pipeline

    report = pipeline.run(
        query=args.query,
        department=args.department,
        model_key=args.model,
    )

    print()
    print("=" * 60)
    print(f"问题: {report.query}")
    print(f"安全等级: {report.safety.risk_level}")
    print(f"置信度: {report.confidence:.0%}")
    print("=" * 60)
    print()
    # Windows GBK 编码安全：替换无法编码的字符
    answer_text = report.answer.encode("gbk", errors="replace").decode("gbk")
    print(answer_text)
    print()

    if report.citations:
        print(f"引用来源 ({len(report.citations)} 条):")
        for i, cite in enumerate(report.citations[:3], 1):
            if isinstance(cite, dict):
                dept = cite.get("metadata", {}).get("department", "未知")
                text = cite.get("text", str(cite))[:100]
                print(f"  [{i}] ({dept}) {text}...")

    if report.safety.flagged_segments:
        print(f"\n安全标记: {len(report.safety.flagged_segments)} 条")

    # 打印执行日志路径
    log_path = report.trace.get("exec_log", "")
    if log_path:
        print(f"\n执行日志: {log_path}")


def cmd_eval(args):
    from src.data.loader import loader as data_loader
    from src.rag.hybrid_retriever import hybrid_retriever

    n = args.sample or 100
    print(f"加载 {n} 条测试数据...")
    test_pairs = data_loader.load_sample(n)

    collection_name = config.get("dataset.collection_name", "med_all")

    # 确保 BM25 索引就绪
    print("构建 BM25 索引...")
    hybrid_retriever._ensure_bm25_index_by_collection(collection_name)

    # 1. 检索评测
    print(f"\n{'='*50}")
    print("1/3 检索评测 (Recall@K, MRR, NDCG)")
    print("=" * 50)

    from src.evaluation.retrieval_metrics import RetrievalEvaluator
    ret_eval = RetrievalEvaluator()
    ret_results = ret_eval.evaluate_with_hybrid(
        test_pairs, hybrid_retriever, collection_name,
    )
    for metric, value in sorted(ret_results.items()):
        if metric != "total_queries":
            print(f"  {metric}: {value:.4f}")

    # 2. 答案质量评测
    print(f"\n{'='*50}")
    print("2/3 答案质量评测 (BLEU, ROUGE-L)")
    print("=" * 50)

    def generate_fn(query):
        from src.pipeline.langgraph_pipeline import langgraph_pipeline as pipeline
        return pipeline.run(query=query).answer

    from src.evaluation.answer_quality import AnswerQualityEvaluator
    ans_eval = AnswerQualityEvaluator()
    ans_results = ans_eval.evaluate_batch(test_pairs[:50], generate_fn)
    for metric, value in sorted(ans_results.items()):
        if metric != "total_pairs":
            print(f"  {metric}: {value:.4f}")

    print("\n评测完成。")


def main():
    parser = argparse.ArgumentParser(description="MedTrust-RAG")
    sub = parser.add_subparsers(dest="command")

    p_idx = sub.add_parser("index", help="索引知识库")
    p_idx.add_argument("--sample", "-s", type=int, default=None)
    p_idx.add_argument("--all", action="store_true")
    p_idx.add_argument("--force", "-f", action="store_true", help="强制重建（删旧重建）")

    p_query = sub.add_parser("query", help="医疗问答")
    p_query.add_argument("query", help="问题文本")
    p_query.add_argument("--model", "-m", default=None)
    p_query.add_argument("--department", "-d", default=None)

    p_eval = sub.add_parser("eval", help="评测")
    p_eval.add_argument("--sample", "-s", type=int, default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config.load(str(PROJECT_ROOT / "config" / "settings.yaml"))

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
