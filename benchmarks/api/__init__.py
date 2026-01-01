"""
LLM API统一封装
支持Gemini、DeepSeek和Claude模型的便捷调用
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseLLMClient
from .gemini import GeminiClient
from .deepseek import DeepSeekClient
from .claude import ClaudeClient


# 模型映射
MODEL_CLASSES = {
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "claude": ClaudeClient,
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    从JSON文件加载配置

    Args:
        config_path: 配置文件路径，默认为 api/config/llm_config.json

    Returns:
        dict: 配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: 配置文件格式错误
    """
    if config_path is None:
        # 默认配置文件路径
        current_dir = Path(__file__).parent
        config_path = current_dir / "config" / "llm_config.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_client(model: str, **config) -> BaseLLMClient:
    """
    工厂函数：创建LLM客户端实例

    Args:
        model: 模型名称（"gemini" 或 "deepseek"）
        **config: 模型特定的配置参数

    Returns:
        BaseLLMClient: 客户端实例

    Raises:
        ValueError: 不支持的模型类型

    示例:
        >>> client = get_client("gemini",
        ...                    project="your-project",
        ...                    location="us-central1")
        >>> result = client.generate("讲个笑话")
    """
    model = model.lower()
    if model not in MODEL_CLASSES:
        raise ValueError(
            f"不支持的模型: {model}. "
            f"支持的模型: {', '.join(MODEL_CLASSES.keys())}"
        )

    client_class = MODEL_CLASSES[model]
    return client_class(**config)


def get_client_from_config(
    model: str,
    config_path: Optional[str] = None
) -> BaseLLMClient:
    """
    从配置文件创建LLM客户端

    Args:
        model: 模型名称（"gemini" 或 "deepseek"）
        config_path: 配置文件路径，默认为 api/config/llm_config.json

    Returns:
        BaseLLMClient: 客户端实例

    Raises:
        ValueError: 配置文件中没有该模型的配置

    示例:
        >>> client = get_client_from_config("gemini")
        >>> result = client.generate("讲个笑话")
    """
    config = load_config(config_path)
    model = model.lower()

    if model not in config:
        raise ValueError(
            f"配置文件中没有找到模型'{model}'的配置. "
            f"可用的模型: {', '.join(config.keys())}"
        )

    model_config = config[model]
    return get_client(model, **model_config)


def batch_generate(
    prompts: List[str],
    model: str,
    max_workers: int = 5,
    show_progress: bool = True,
    config_path: Optional[str] = None,
    **config
) -> List[Dict[str, Any]]:
    """
    批量生成文本（支持并发）

    Args:
        prompts: 提示词列表
        model: 模型名称（"gemini" 或 "deepseek"）
        max_workers: 最大并发线程数，默认5
        show_progress: 是否显示进度条，默认True
        config_path: 配置文件路径（如果提供，优先使用配置文件）
        **config: 模型配置参数（如果不使用配置文件）

    Returns:
        List[Dict]: 结果列表，每个元素包含:
            - prompt: 原始提示词
            - result: 生成的文本（成功时）
            - error: 错误信息（失败时）
            - success: 是否成功

    示例:
        >>> # 使用配置文件
        >>> results = batch_generate(
        ...     prompts=["问题1", "问题2", "问题3"],
        ...     model="gemini",
        ...     max_workers=3
        ... )

        >>> # 直接传递配置
        >>> results = batch_generate(
        ...     prompts=["问题1", "问题2"],
        ...     model="deepseek",
        ...     api_key="your-key",
        ...     appid="your-appid"
        ... )
    """
    # 创建客户端
    if config_path:
        client = get_client_from_config(model, config_path)
    else:
        client = get_client(model, **config)

    # 调用客户端的batch_generate方法
    return client.batch_generate(
        prompts=prompts,
        max_workers=max_workers,
        show_progress=show_progress
    )


# 导出所有公共接口
__all__ = [
    # 类
    "BaseLLMClient",
    "GeminiClient",
    "DeepSeekClient",
    "ClaudeClient",
    # 函数
    "get_client",
    "get_client_from_config",
    "batch_generate",
    "load_config",
]
