"""LangChain链定义 - Prompt从YAML加载"""

import json
import re
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils.logger import logger

PROMPTS_FILE = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"

# 缓存已加载的prompts
_prompts_cache = None


def _load_prompts() -> dict:
    """加载prompts.yaml并缓存"""
    global _prompts_cache
    if _prompts_cache is None:
        if not PROMPTS_FILE.exists():
            raise FileNotFoundError(f"Prompt配置不存在: {PROMPTS_FILE}")
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            _prompts_cache = yaml.safe_load(f) or {}
    return _prompts_cache


def load_prompt_template(prompt_name: str) -> str:
    """从prompts.yaml加载指定prompt模板"""
    prompts = _load_prompts()
    if prompt_name not in prompts:
        raise FileNotFoundError(f"Prompt模板不存在: {prompt_name}")
    return prompts[prompt_name]


def create_review_chain(prompt_name: str, llm: ChatOpenAI) -> any:
    """创建审查链（StrOutputParser + 自定义JSON解析）"""
    from langchain_core.output_parsers import StrOutputParser

    template = load_prompt_template(prompt_name)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    logger.info(f"创建链: {prompt_name}")
    return chain


def parse_json_from_text(text: str) -> dict:
    """从LLM输出文本中解析JSON，处理markdown代码块和中文引号"""
    cleaned = text.strip()
    if not cleaned:
        return {"issues": [], "summary": "LLM返回空内容"}

    # 清理markdown代码块
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # 直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 替换中文引号后解析
    normalized = cleaned
    for old, new in [("\u201c", '\\"'), ("\u201d", '\\"'), ("\u2018", "'"), ("\u2019", "'")]:
        normalized = normalized.replace(old, new)

    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    # 正则提取JSON对象
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            try:
                return json.loads(json_match.group().replace("\u201c", '\\"').replace("\u201d", '\\"'))
            except json.JSONDecodeError:
                pass

    logger.warning(f"JSON解析失败, 原始文本前200字符: {cleaned[:200]}")
    return {"issues": [], "summary": "JSON解析失败"}
