"""Query API routes"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException

from app.schemas import (
    QueryRequest,
    QueryResponse,
    Citation,
    SafetyInfo,
    ModelInfo,
    DepartmentInfo,
)
from src.pipeline.langgraph_pipeline import langgraph_pipeline as pipeline
from src.utils.config_loader import config
from src.rag.hybrid_retriever import hybrid_retriever

router = APIRouter()


def get_available_models() -> list[str]:
    available = []
    try:
        models_cfg = config.get("llm.models", {})
        for name, cfg in models_cfg.items():
            api_key = cfg.get("api_key", "")
            if api_key and not api_key.startswith("${"):
                available.append(name)
            elif api_key.startswith("${") and api_key.endswith("}"):
                import os
                env_var = api_key[2:-1]
                if os.environ.get(env_var):
                    available.append(name)
    except Exception:
        pass
    return available or ["zhipu"]


def get_model_display_name(model_key: str) -> str:
    """返回模型的展示名称（如 GLM-4.5-Air），而非配置 key（如 zhipu）"""
    try:
        return config.get(f"llm.models.{model_key}.model_name", model_key)
    except Exception:
        return model_key


def get_departments() -> list[str]:
    try:
        from src.rag.vector_store import vector_store
        collection = vector_store.client.get_collection("med_all")
        total = collection.count()
        all_depts = set()
        for offset in range(0, total, 999):
            batch = collection.get(include=["metadatas"], offset=offset, limit=999)
            for m in batch.get("metadatas", []):
                dept = m.get("department", "")
                if dept:
                    all_depts.add(dept)
        return sorted(all_depts)[:10]
    except Exception:
        return []


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    return [ModelInfo(name=m) for m in get_available_models()]


@router.get("/departments", response_model=list[DepartmentInfo])
async def list_departments():
    return [DepartmentInfo(name=d) for d in get_departments()]


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "bm25_ready": "med_all" in hybrid_retriever._bm25_indices,
        "available_models": get_available_models(),
    }


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # 空查询校验
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="查询内容不能为空")

    # 无效模型校验
    if req.model and req.model not in get_available_models():
        raise HTTPException(status_code=400, detail=f"不支持的模型: {req.model}，可选: {get_available_models()}")

    # 模型选择
    default_model = config.get("llm.default_model", "zhipu")
    model_key = req.model if req.model in get_available_models() else default_model

    # 科室
    dept = req.department if req.department and req.department != "全部科室" else None

    # 执行 pipeline（async，避免 asyncio.run 冲突）
    report = await pipeline.arun(
        query=req.query.strip(),
        department=dept,
        model_key=model_key,
    )

    # 转换引用
    citations = []
    for c in (report.citations or []):
        if isinstance(c, dict):
            citations.append(Citation(
                text=c.get("text", ""),
                department=c.get("metadata", {}).get("department"),
                source_id=str(c.get("metadata", {}).get("source_id", "")) or None,
            ))
        else:
            citations.append(Citation(text=str(c)))

    return QueryResponse(
        query=req.query,
        answer=report.answer,
        confidence=report.confidence,
        model_used=get_model_display_name(report.model_used or model_key),
        safety=SafetyInfo(
            risk_level=report.safety.risk_level,
            flagged_segments=report.safety.flagged_segments or [],
            suggestions=report.safety.suggestions or [],
            contradictions=report.safety.contradictions or [],
        ),
        citations=citations,
        trace=report.trace,
    )
