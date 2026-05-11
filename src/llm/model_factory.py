"""模型工厂 - 多模型切换，统一接口"""

from langchain_openai import ChatOpenAI

from src.utils.config_loader import config
from src.utils.logger import logger


class ModelFactory:
    """LLM模型工厂，根据配置创建不同模型实例"""

    @staticmethod
    def create_chat_model(model_key: str = None, **kwargs) -> ChatOpenAI:
        """创建Chat模型实例"""
        return ModelFactory._create_model("llm", model_key, **kwargs)

    @staticmethod
    def create_multimodal_model(model_key: str = None, **kwargs) -> ChatOpenAI:
        """创建多模态模型实例（用于OCR降级）"""
        return ModelFactory._create_model("multimodal_llm", model_key, **kwargs)

    @staticmethod
    def _create_model(config_key: str, model_key: str = None, **kwargs) -> ChatOpenAI:
        """通用模型创建"""
        if model_key is None:
            model_key = config.get(f"{config_key}.default_model", "deepseek")

        models_config = config.get(f"{config_key}.models", {})
        if model_key not in models_config:
            raise ValueError(f"未找到模型配置: {model_key}，可用模型: {list(models_config.keys())}")

        model_cfg = models_config[model_key]

        api_key = model_cfg.get("api_key", "")
        if not api_key:
            raise ValueError(
                f"模型 {model_key} 的 api_key 未配置。"
                f"请在 config/settings.yaml 中设置 api_key 或配置环境变量。"
            )

        params = {
            "model": model_cfg.get("model_name"),
            "api_key": api_key,
            "base_url": model_cfg.get("api_base"),
            "temperature": model_cfg.get("temperature", 0.1),
            "max_tokens": model_cfg.get("max_tokens", 4096),
        }
        params.update(kwargs)

        logger.info(f"创建模型: {model_key} ({params['model']})")
        return ChatOpenAI(**params)

    @staticmethod
    def list_available_models() -> dict:
        """列出所有可用模型"""
        return {
            "chat_models": list(config.get("llm.models", {}).keys()),
            "multimodal_models": list(config.get("multimodal_llm.models", {}).keys()),
        }
