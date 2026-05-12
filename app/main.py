"""MedTrust-RAG FastAPI 入口"""

import sys
import os
from pathlib import Path

# 强制 HuggingFace 使用本地缓存，跳过联网检查（reranker 模型已缓存）
os.environ["HF_HUB_OFFLINE"] = "1"

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.config_loader import config
from app.routers import query as query_router

config.load(str(project_root / "config" / "settings.yaml"))

app = FastAPI(title="MedTrust-RAG", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(query_router.router, prefix="", tags=["query"])


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(Path(__file__).parent / "templates" / "index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
