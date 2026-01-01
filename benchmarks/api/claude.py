"""
Claude API客户端实现
基于Anthropic官方SDK
"""
from typing import Optional
from anthropic import Anthropic
from .base import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """
    Claude API客户端

    示例:
        >>> client = ClaudeClient(
        ...     api_key="your-api-key",
        ...     model_name="claude-sonnet-4-20250514"
        ... )
        >>> response = client.generate("讲个笑话")
    """

    def _setup(self):
        """初始化Claude客户端"""
        # 从配置中获取参数
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "claude-sonnet-4-20250514")
        self.base_url = self.config.get("base_url")  # 可选，用于代理
        self.default_max_tokens = self.config.get("max_new_tokens", 1024)
        self.default_temperature = self.config.get("temperature", 1.0)

        # 验证必需参数
        if not self.api_key:
            raise ValueError("api_key是必需参数")

        # 初始化Anthropic客户端
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = Anthropic(**client_kwargs)

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        调用Claude API生成文本

        Args:
            prompt: 输入提示词
            temperature: 温度参数（0.0-1.0），默认1.0
            max_tokens: 最大生成token数，默认1024
            **kwargs: 其他Claude特定参数，如:
                - system: 系统提示词
                - top_p: 核采样参数
                - top_k: top-k采样参数

        Returns:
            str: 生成的文本内容

        Raises:
            Exception: API调用失败时抛出异常
        """
        # 设置默认值（优先使用配置文件中的值）
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # 提取系统提示词（如果有）
        system = kwargs.pop("system", None)

        # 准备请求参数
        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        # 添加可选参数
        if temperature is not None:
            request_params["temperature"] = temperature
        if system:
            request_params["system"] = system

        # 添加其他自定义参数（如top_p, top_k等）
        for key in ["top_p", "top_k", "stop_sequences"]:
            if key in kwargs:
                request_params[key] = kwargs.pop(key)

        # 调用API
        response = self.client.messages.create(**request_params)

        # 提取生成的文本
        if response and response.content:
            # Claude返回的content是一个列表，通常第一个元素是文本
            text_blocks = [
                block.text for block in response.content
                if hasattr(block, 'text')
            ]
            if text_blocks:
                return "".join(text_blocks)
            else:
                raise Exception("API返回空响应")
        else:
            raise Exception("API返回无效响应")
