"""MedTrust-RAG Streamlit UI"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from src.utils.config_loader import config
from src.data.loader import loader as data_loader


def init():
    config.load(str(project_root / "config" / "settings.yaml"))
    st.set_page_config(
        page_title="MedTrust-RAG",
        page_icon="🏥",
        layout="wide",
    )


def render_sidebar():
    with st.sidebar:
        st.title("MedTrust-RAG")
        st.caption("医疗可信问答系统")

        st.divider()

        model = st.selectbox(
            "LLM 模型",
            options=["deepseek", "zhipu", "qwen"],
            index=0,
        )

        try:
            departments = ["全部科室"] + data_loader.get_departments()[:10]
        except Exception:
            departments = ["全部科室"]
        dept = st.selectbox("科室过滤", options=departments)
        if dept == "全部科室":
            dept = None

        st.divider()

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

    with st.spinner("正在检索医学知识..."):
        from src.pipeline.medical_pipeline import pipeline

        report = pipeline.run(
            query=query.strip(),
            department=department,
            model_key=model_key,
        )

    st.divider()

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

    if report.citations:
        with st.expander(f"查看引用来源（{len(report.citations)} 条）"):
            for i, cite in enumerate(report.citations[:5], 1):
                if isinstance(cite, dict):
                    dept = cite.get("metadata", {}).get("department", "未知科室")
                    text = cite.get("text", str(cite))
                    st.markdown(f"**[{i}] 科室: {dept}**")
                    st.text(text[:300])

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

    if report.trace:
        with st.expander("处理追踪"):
            st.json(report.trace)


if __name__ == "__main__":
    init()
    if "query" not in st.session_state:
        st.session_state.query = ""
    model_key, dept = render_sidebar()
    render_main(model_key, dept)
