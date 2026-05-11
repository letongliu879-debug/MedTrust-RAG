"""配置加载器 - 从YAML文件加载配置，支持环境变量替换"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """递归解析配置中的环境变量占位符 ${VAR_NAME}"""
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)
        for match in matches:
            env_val = os.environ.get(match, "")
            value = value.replace(f"${{{match}}}", env_val)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class ConfigLoader:
    """全局配置加载器，单例模式"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = None) -> dict:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self._config = _resolve_env_vars(self._config)
        return self._config

    def get(self, key_path: str = None, default: Any = None) -> Any:
        """
        获取配置项，支持点号路径
        例: config.get("llm.default_model") -> "deepseek"
        """
        if self._config is None:
            self.load()

        if key_path is None:
            return self._config

        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def reload(self, config_path: str = None) -> dict:
        """重新加载配置"""
        self._config = None
        return self.load(config_path)


# 全局配置实例
config = ConfigLoader()
