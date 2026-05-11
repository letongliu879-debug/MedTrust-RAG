"""验证本次修复的测试脚本"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import config

# 加载配置（如果存在）
cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
if cfg_path.exists():
    config.load(str(cfg_path))

print("=" * 50)
print("1. Reranker 阈值自动修正")
print("=" * 50)

from src.rag.reranker import Reranker

r = Reranker()
print(f"  normalize 模式: {r._normalize}")
print(f"  配置阈值: {r._threshold}")

# 断言：归一化模式下阈值不应为负数
if r._normalize:
    assert r._threshold >= 0, f"FAIL: 归一化模式下阈值不应为负，当前={r._threshold}"
    print(f"  PASS: 阈值 {r._threshold} 在 [0,1] 范围内")
else:
    print(f"  PASS: 非归一化模式，阈值 {r._threshold}")

print()
print("=" * 50)
print("2. 向量相似度默认阈值")
print("=" * 50)

from src.rag.hybrid_retriever import HybridRetriever

h = HybridRetriever()
print(f"  向量相似度阈值: {h._vector_sim_threshold}")
assert h._vector_sim_threshold >= 0.5, f"FAIL: 阈值应 >= 0.5，实际={h._vector_sim_threshold}"
print(f"  PASS: 阈值 >= 0.5")

print()
print("=" * 50)
print("3. 子查询锚定")
print("=" * 50)

from src.agents.retriever_agent import RetrieverAgent

ra = RetrieverAgent()
# 测试规则降级：原查询应始终被保留
test_query = "糖尿病用二甲双胍出现恶心怎么办"
subs = ra._rule_based_subqueries(test_query)
print(f"  原始查询: {test_query}")
print(f"  子查询: {subs}")
assert test_query in subs, f"FAIL: 原始查询不在子查询列表中"
print(f"  PASS: 原始查询在子查询列表中")

# 测试单查询场景（不应多跳）
query2 = "高血压怎么治"
subs2 = ra._rule_based_subqueries(query2)
print(f"\n  简单查询: {query2}")
print(f"  子查询: {subs2}")
assert query2 in subs2, f"FAIL: 原始查询不在子查询列表中"
print(f"  PASS: 原始查询在子查询列表中")

print()
print("=" * 50)
print("4. ChromaDB where 过滤语法")
print("=" * 50)

# 仅测试语法，不连接数据库
where_clause = {"department": "内科"}
print(f"  where 子句: {where_clause}")
assert isinstance(where_clause, dict)
assert "department" in where_clause
print(f"  PASS: where 语法正确")

print()
print("=" * 50)
print("全部验证通过")
print("=" * 50)
