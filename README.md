# MedTrust-RAG

医疗可信问答系统。基于 **Huatuo26M-Lite**（~178K 中文医疗 QA 对）构建，采用**混合检索 + 多 Agent 交叉验证**架构，确保回答的准确性与安全性。

## 核心特性

- **混合检索**：BM25 关键词 + BGE-M3 向量语义双路检索，RRF 融合
- **BM25 三级加载**：内存缓存 → 磁盘 pickle 缓存 → ChromaDB tokens 预分词字段（跳过 jieba）→ 全量 jieba 分词降级，冷启动从 ~196s 降至 ~1-2s
- **智能子查询**：LLM 自动判断多跳问题，并行生成多个检索子查询
- **BGE-Reranker 精排**：交叉编码器重排序，提升相关文档召回
- **四 Agent 管线**：Retriever → Responder → SafetyChecker → Synthesizer 串行校验，带实时进度反馈
- **医疗安全校验**：幻觉检测 + 危险建议识别 + 证据交叉验证

## 架构

```
用户问题
  │
  ▼
┌─────────────────────────────┐
│   RetrieverAgent            │
│  ┌─────────────────────────┐│
│  │ 多跳判断 + 子查询生成    ││
│  └─────────────────────────┘│
│  ┌───────────┬─────────────┐│
│  │  BM25    │   向量检索   ││
│  └───────────┴─────────────┘│
│         ↓ RRF 融合          │
│  ┌─────────────────────────┐│
│  │   BGE-Reranker 精排     ││
│  └─────────────────────────┘│
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   ResponderAgent            │
│   基于证据生成答案 + 引用    │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   SafetyCheckerAgent        │
│   幻觉检测 + 安全校验       │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│   SynthesizerAgent          │
│   风险分级 + 答案精炼       │
└─────────────────────────────┘
  │
  ▼
最终答案 {answer, citations, confidence, safety}
```

## 数据

`FreedomIntelligence/Huatuo26M-Lite`（HuggingFace），Apache-2.0 许可。

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/)（本地运行 BGE-M3 Embedding）
- LLM API（支持 OpenAI 兼容接口）：Zhipu GLM-4 / DeepSeek / Qwen

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 拉取 Ollama Embedding 模型
ollama pull bge-m3

# 配置文件
cp config/settings.example.yaml config/settings.yaml
# 编辑 settings.yaml 填入 API Key
```

## 索引构建

```bash
# 全量索引（约 178K 条）
python main.py index --all

# 采样索引（测试用）
python main.py index --sample 1000

# 按科室分别建索引
python main.py index --by-department

# 强制重建
python main.py index --all --force
```

## 使用

```bash
# 命令行问答
python main.py query "全身酸痛是不是流感"

# Web UI
streamlit run app/app.py
```

## 配置

配置文件 `config/settings.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm.default_model` | 默认 LLM | zhipu |
| `embedding.model_name` | Embedding 模型 | bge-m3 |
| `rag.retrieval_top_k` | 最终召回数量 | 5 |
| `rag.hybrid.bm25_top_k` | BM25 召回数 | 20 |
| `rag.hybrid.vector_top_k` | 向量召回数 | 10 |
| `rag.hybrid.vector_similarity_threshold` | 向量相似度阈值 | 0.4 |
| `rag.reranker.threshold` | Reranker 过滤阈值 | 0.3 |

## 项目结构

```
├── app/                    # Streamlit Web UI
├── config/                 # 配置文件 & Prompt 模板
│   ├── settings.yaml
│   └── prompts.yaml
├── data/                   # ChromaDB 持久化 & 日志
├── scripts/                # 工具脚本（索引构建等）
├── src/
│   ├── agents/            # 4 个 Agent 实现
│   │   ├── retriever_agent.py
│   │   ├── responder_agent.py
│   │   ├── safety_checker.py
│   │   └── synthesizer.py
│   ├── data/              # 数据加载
│   ├── evaluation/        # 评测模块
│   ├── llm/               # LLM 模型 & Chain 封装
│   ├── pipeline/          # 主流程编排
│   ├── rag/               # 检索核心
│   │   ├── hybrid_retriever.py   # 混合检索（BM25+向量+RRF）
│   │   ├── reranker.py           # BGE-Reranker 精排
│   │   ├── vector_store.py       # ChromaDB 管理
│   │   └── embeddings.py         # Embedding 模型
│   └── utils/             # 工具函数
├── main.py                # CLI 入口
└── requirements.txt
```

## 评测体系

- **检索**：Recall@K、MRR、NDCG@5
- **答案质量**：BLEU-4、ROUGE-L、BERTScore
- **LLM 裁判**：忠实度 / 安全性 / 相关性 / 完整性
