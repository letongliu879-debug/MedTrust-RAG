# MedTrust-RAG

基于 Huatuo26M-Lite（~178K 中文医疗 QA 对）+ 混合检索 + 多 Agent 交叉验证的医疗可信问答系统。

## 检索架构

```
用户问题
  → RetrieverAgent（LLM 子查询生成 + 混合检索 BM25+向量 RRF → BGE-Reranker 精排）
  → ResponderAgent（基于证据生成答案 + 引用来源）
  → SafetyCheckerAgent（幻觉检测 + 医疗安全校验 + 证据交叉验证）
  → SynthesizerAgent（按风险等级精炼最终答案 + 免责声明）
  → 最终答案 {answer, citations, confidence, safety_notes}
```

## 数据

`FreedomIntelligence/Huatuo26M-Lite` (HuggingFace)，Apache-2.0 许可。

## 快速开始

```bash
pip install -r requirements.txt
ollama pull bge-m3（使用本地ollama的embedding模型）

cp config/settings.example.yaml config/settings.yaml
# 编辑 settings.yaml 填入 API Key

# 构建持久化的向量数据库
python main.py index --all

# 问答
python main.py query "高血压怎么治疗"

# Web UI
streamlit run app/app.py
```

## 评测体系

- **检索**: Recall@K, MRR, NDCG@5（对照 ground truth QA 对 ID）
- **答案质量**: BLEU-4, ROUGE-L, BERTScore（vs Huatuo-Lite 参考答案）
- **LLM 裁判**: 忠实度 / 安全性 / 相关性 / 完整性

## 项目结构

```
├── app/              # Streamlit UI
├── src/
│   ├── data/         # 数据加载 + 索引
│   ├── rag/          # 混合检索 + 重排序 + 向量存储
│   ├── agents/       # 4 Agent（Retriever/Responder/Safety/Synthesizer）
│   ├── pipeline/     # 主流程编排
│   ├── llm/          # LLM 模型 & Chain
│   ├── evaluation/   # 检索/答案质量/LLM裁判 评测
│   └── utils/        # 配置加载 & 日志
├── config/           # 配置 & Prompt
├── main.py           # CLI
└── data/             # ChromaDB 持久化
```
