"""
Gemini API客户端实现
基于Google Vertex AI的Gemini模型
"""
import os
from typing import Optional
from vertexai.generative_models import GenerativeModel
import vertexai
from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Gemini API客户端

    示例:
        >>> client = GeminiClient(
        ...     project="your-project",
        ...     location="us-central1",
        ...     model_name="gemini-2.5-pro",
        ...     credentials_path="path/to/credentials.json"
        ... )
        >>> response = client.generate("讲个笑话")
    """

    def _setup(self):
        """初始化Gemini客户端"""
        # 从配置中获取参数
        self.project = self.config.get("project")
        self.location = self.config.get("location")
        self.model_name = self.config.get("model_name", "gemini-2.5-pro")
        credentials_path = self.config.get("credentials_path")
        self.default_max_tokens = self.config.get("max_new_tokens")  # 无默认值
        self.default_temperature = self.config.get("temperature")  # 无默认值

        # 设置凭证路径
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        # 验证必需参数
        if not self.project or not self.location:
            raise ValueError("project和location是必需参数")

        # 初始化Vertex AI
        vertexai.init(project=self.project, location=self.location)
        self.model = GenerativeModel(self.model_name)

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        调用Gemini API生成文本

        Args:
            prompt: 输入提示词
            temperature: 温度参数（0.0-1.0）
            max_tokens: 最大生成token数
            **kwargs: 其他Gemini特定参数

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

        # 准备生成配置
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        # 调用API
        if generation_config:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
        else:
            response = self.model.generate_content(prompt)

        # 返回生成的文本
        if response and response.text:
            return response.text
        else:
            raise Exception("API返回空响应")
