"""Huatuo26M-Lite 数据集加载器

从 HuggingFace 加载精简版中文医疗 QA 数据集。
数据集路径: FreedomIntelligence/Huatuo26M-Lite
字段: id, question, answer, label(department), score, related_diseases
"""

from src.utils.config_loader import config
from src.utils.logger import logger


class HuatuoLoader:
    """加载和预处理 Huatuo26M-Lite 数据集"""

    def __init__(self):
        self._dataset_name = config.get(
            "dataset.name",
            "FreedomIntelligence/Huatuo26M-Lite",
        )
        self._dataset = None

    # ============ 加载 ============

    def load(self, split: str = "train") -> list[dict]:
        """全量加载。返回 [{"id":..., "question":..., "answer":..., ...}, ...]"""
        ds = self._load_hf_dataset(split)
        records = []
        for item in ds:
            records.append({
                "id": int(item.get("id", 0)),
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "department": str(item.get("label", "")),
                "score": int(item.get("score", 0)),
                "related_diseases": str(item.get("related_diseases", "")),
            })
        logger.info(f"加载 Huatuo-Lite: {len(records)} 条")
        return records

    def load_sample(self, n: int = 500) -> list[dict]:
        """加载抽样数据（用于快速测试）"""
        ds = self._load_hf_dataset("train")
        records = []
        for i, item in enumerate(ds):
            if i >= n:
                break
            records.append({
                "id": int(item.get("id", 0)),
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "department": str(item.get("label", "")),
                "score": int(item.get("score", 0)),
                "related_diseases": str(item.get("related_diseases", "")),
            })
        logger.info(f"加载 Huatuo-Lite 抽样: {len(records)} 条")
        return records

    def get_departments(self) -> list[str]:
        """返回所有科室列表（按数据量降序）"""
        ds = self._load_hf_dataset("train")
        dept_count: dict[str, int] = {}
        for item in ds:
            dept = str(item.get("label", "")).strip()
            if dept:
                dept_count[dept] = dept_count.get(dept, 0) + 1
        return sorted(dept_count, key=dept_count.get, reverse=True)

    def get_diseases(self) -> list[str]:
        """返回所有疾病标签"""
        ds = self._load_hf_dataset("train")
        diseases = set()
        for item in ds:
            rd = str(item.get("related_diseases", "")).strip()
            if rd:
                diseases.add(rd)
        return sorted(diseases)

    # ============ 内部 ============

    def _load_hf_dataset(self, split: str):
        if self._dataset is None:
            try:
                from datasets import load_dataset
                self._dataset = load_dataset(self._dataset_name, split=split)
                logger.info(f"从 HuggingFace 加载数据集: {self._dataset_name}")
            except ImportError:
                raise ImportError(
                    "需要安装 datasets 库: pip install datasets"
                )
            except Exception as e:
                logger.error(f"加载数据集失败: {e}")
                raise
        return self._dataset


# 全局实例
loader = HuatuoLoader()
