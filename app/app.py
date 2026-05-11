"""MedTrust-RAG Streamlit UI — 带实时进度反馈"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from src.utils.config_loader import config


# ============ 初始化 ============

def init():
    config.load(str(project_root / "config" / "settings.yaml"))
    st.set_page_config(
        page_title="MedTrust-RAG",
        page_icon="🏥",
        layout="wide",
    )
    # 自动后台预热 BM25 索引（不阻塞 UI）
    if "bm25_warmup_started" not in st.session_state:
        import threading
        from src.rag.hybrid_retriever import hybrid_retriever
        st.session_state.bm25_warmup_started = True
        st.session_state.bm25_ready = "med_all" in hybrid_retriever._bm25_indices

        def _warmup():
            hybrid_retriever._ensure_bm25_index("med_all")
            st.session_state.bm25_ready = True

        if not st.session_state.bm25_ready:
            t = threading.Thread(target=_warmup, daemon=True)
            t.start()



def get_available_models() -> list[str]:
    """获取已配置有效 api_key 的模型列表"""
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


def get_departments() -> list[str]:
    """从 ChromaDB metadata 获取科室列表"""
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
        return ["全部科室"] + sorted(all_depts)[:10]
    except Exception:
        return ["全部科室"]


# ============ 侧边栏 ============

def render_sidebar():
    with st.sidebar:
        st.title("MedTrust-RAG")
        st.caption("医疗可信问答系统")

        st.divider()

        # 模型选择
        available = get_available_models()
        default_model = config.get("llm.default_model", "zhipu")
        if default_model not in available:
            default_model = available[0]
        model = st.selectbox(
            "LLM 模型",
            options=available,
            index=available.index(default_model) if default_model in available else 0,
        )

        # 科室过滤
        departments = get_departments()
        dept = st.selectbox("科室过滤", options=departments)
        if dept == "全部科室":
            dept = None

        # BM25 索引状态
        st.divider()
        with st.expander("系统状态"):
            if st.session_state.get("bm25_ready"):
                st.info("BM25 索引: ✅ 已就绪")
            else:
                st.warning("BM25 索引: ⏳ 后台构建中（首次查询可能较慢）")

        st.divider()

        # 示例问题
        st.markdown("**示例问题**")
        examples = [
            "高血压患者日常饮食需要注意什么？",
            "婴儿发烧38度应该怎么处理？",
            "糖尿病可以根治吗？",
            "颈椎病有什么症状，怎么缓解？",
            "怀孕初期出现腹痛正常吗？",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.query = ex

        st.divider()
        st.caption("数据来源: Huatuo26M-Lite")
        st.caption("约 17.8 万条中文医疗 QA 对")

    return model, dept


# ============ 主界面 ============

# 阶段定义
STAGES = [
    ("retrieve", "🔍 检索证据", "Retriever"),
    ("responder", "💡 生成答案", "Responder"),
    ("safety", "🛡️ 安全校验", "Safety Checker"),
    ("synthesize", "📝 合成答案", "Synthesizer"),
]


def render_progress_panel(progress_data: dict):
    """渲染实时进度面板"""
    for stage_key, stage_label, stage_name in STAGES:
        info = progress_data.get(stage_key, {})
        status = info.get("status", "pending")  # pending / running / done

        if status == "pending":
            st.markdown(f"⏳ **{stage_label}** — 等待中")
        elif status == "running":
            elapsed = info.get("elapsed", 0)
            st.markdown(f"🔄 **{stage_label}** — 执行中 ({elapsed:.1f}s)")
        elif status == "done":
            elapsed = info.get("elapsed", 0)
            detail = info.get("detail", "")
            st.markdown(f"✅ **{stage_label}** — 完成 ({elapsed:.1f}s) {detail}")


def render_main(model_key: str, department: str = None):
    st.title("医疗问答")

    query = st.text_area(
        "输入您的健康问题",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="例如：高血压应该怎么控制？有哪些注意事项？",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search = st.button("查询", type="primary", use_container_width=True)

    if not search and not query:
        st.info("请输入健康问题，点击查询开始")
        return

    if not query.strip():
        st.warning("请输入有效的问题")
        return

    # ---- 执行查询 ----
    from src.pipeline.medical_pipeline import pipeline

    # 进度数据：记录每个阶段的状态
    progress_data = {
        "retrieve": {"status": "pending"},
        "responder": {"status": "pending"},
        "safety": {"status": "pending"},
        "synthesize": {"status": "pending"},
    }
    stage_start_times = {}

    def on_progress(step: str, meta: dict):
        """进度回调：更新 progress_data"""
        if step == "retrieve_start":
            progress_data["retrieve"]["status"] = "running"
            stage_start_times["retrieve"] = time.time()
        elif step == "retrieve_done":
            progress_data["retrieve"]["status"] = "done"
            progress_data["retrieve"]["elapsed"] = time.time() - stage_start_times.get("retrieve", time.time())
            progress_data["retrieve"]["detail"] = f"检索到 {meta.get('chunks', 0)} 条证据"
        elif step == "responder_start":
            progress_data["responder"]["status"] = "running"
            stage_start_times["responder"] = time.time()
        elif step == "responder_done":
            progress_data["responder"]["status"] = "done"
            progress_data["responder"]["elapsed"] = time.time() - stage_start_times.get("responder", time.time())
            progress_data["responder"]["detail"] = f"置信度 {meta.get('confidence', 0):.0%}"
        elif step == "safety_start":
            progress_data["safety"]["status"] = "running"
            stage_start_times["safety"] = time.time()
        elif step == "safety_done":
            progress_data["safety"]["status"] = "done"
            progress_data["safety"]["elapsed"] = time.time() - stage_start_times.get("safety", time.time())
            risk = meta.get("risk_level", "safe")
            flagged = meta.get("flagged", 0)
            progress_data["safety"]["detail"] = f"风险等级: {risk}" + (f"，标记 {flagged} 项" if flagged else "")
        elif step == "synthesize_start":
            progress_data["synthesize"]["status"] = "running"
            stage_start_times["synthesize"] = time.time()
        elif step == "synthesize_done":
            progress_data["synthesize"]["status"] = "done"
            progress_data["synthesize"]["elapsed"] = time.time() - stage_start_times.get("synthesize", time.time())
            progress_data["synthesize"]["detail"] = f"最终置信度 {meta.get('confidence', 0):.0%}"

    # 用 status 容器展示进度
    with st.status("正在处理您的健康问题...", expanded=True) as status:
        progress_placeholder = st.empty()
        total_start = time.time()

        # 在子线程中运行 pipeline，同时轮询进度
        import threading

        report_result = [None]
        report_error = [None]

        def run_pipeline():
            try:
                report_result[0] = pipeline.run(
                    query=query.strip(),
                    department=department,
                    model_key=model_key,
                    on_progress=on_progress,
                )
            except Exception as e:
                report_error[0] = e

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        # 轮询进度
        while thread.is_alive():
            elapsed_total = time.time() - total_start
            # 更新正在运行的阶段的 elapsed
            for stage_key in stage_start_times:
                if progress_data[stage_key]["status"] == "running":
                    progress_data[stage_key]["elapsed"] = time.time() - stage_start_times[stage_key]

            with progress_placeholder.container():
                render_progress_panel(progress_data)
                st.caption(f"总耗时: {elapsed_total:.1f}s")

            time.sleep(0.5)

        thread.join()

        # 检查错误
        if report_error[0]:
            status.update(label="处理失败", state="error")
            st.error(f"处理出错: {report_error[0]}")
            return

    report = report_result[0]
    if report is None:
        st.error("处理失败，未获得结果")
        return

    # ---- 结果展示 ----
    status.update(label="处理完成", state="complete", expanded=False)

    # 安全等级
    risk = report.safety.risk_level
    if risk == "safe":
        st.success("安全等级：可信")
    elif risk == "caution":
        st.warning("安全等级：需注意")
    else:
        st.error("安全等级：有风险")

    st.markdown("### 回答")
    st.markdown(report.answer)

    st.caption(f"置信度: {report.confidence:.0%}  |  模型: {report.model_used}")

    # 引用来源
    if report.citations:
        with st.expander(f"查看引用来源（{len(report.citations)} 条）"):
            for i, cite in enumerate(report.citations[:5], 1):
                if isinstance(cite, dict):
                    dept = cite.get("metadata", {}).get("department", "未知科室")
                    text = cite.get("text", str(cite))
                    st.markdown(f"**[{i}] 科室: {dept}**")
                    st.text(text[:300])

    # 安全校验详情
    if report.safety.flagged_segments:
        with st.expander("安全校验详情"):
            if report.safety.flagged_segments:
                st.markdown("**标记的片段:**")
                for item in report.safety.flagged_segments:
                    st.warning(item)
            if report.safety.suggestions:
                st.markdown("**改进建议:**")
                for item in report.safety.suggestions:
                    st.info(item)
            if report.safety.contradictions:
                st.markdown("**证据矛盾:**")
                for item in report.safety.contradictions:
                    st.warning(item)

    # 耗时 trace
    if report.trace:
        with st.expander("处理追踪"):
            st.json(report.trace)


# ============ 入口 ============

if __name__ == "__main__":
    init()
    if "query" not in st.session_state:
        st.session_state.query = ""
    model_key, dept = render_sidebar()
    render_main(model_key, dept)
