"""
DeepSeek API客户端实现
通过百度千帆平台调用DeepSeek模型
"""
from typing import Optional
from openai import OpenAI
from .base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API客户端（通过百度千帆平台）

    示例:
        >>> client = DeepSeekClient(
        ...     api_key="your-api-key",
        ...     base_url="https://qianfan.baidubce.com/v2",
        ...     model_name="deepseek-r1",
        ...     appid="your-appid"
        ... )
        >>> response = client.generate("讲个笑话")
    """

    def _setup(self):
        """初始化DeepSeek客户端"""
        # 从配置中获取参数
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url", "https://qianfan.baidubce.com/v2")
        self.model_name = self.config.get("model_name", "deepseek-r1")
        self.appid = self.config.get("appid")
        self.default_max_tokens = self.config.get("max_new_tokens", 300)
        self.default_temperature = self.config.get("temperature", 0.7)

        # 验证必需参数
        if not self.api_key:
            raise ValueError("api_key是必需参数")
        if not self.appid:
            raise ValueError("appid是必需参数")

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"appid": self.appid}
        )

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        调用DeepSeek API生成文本

        Args:
            prompt: 输入提示词
            temperature: 温度参数（0.0-2.0），默认从配置读取或0.7
            max_tokens: 最大生成token数，默认从配置读取或300
            **kwargs: 其他DeepSeek特定参数

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

        # 准备请求参数
        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        # 添加其他自定义参数
        request_params.update(kwargs)

        # 调用API
        response = self.client.chat.completions.create(**request_params)

        # 提取生成的文本
        if response and response.choices:
            content = response.choices[0].message.content
            if content:
                return content
            else:
                raise Exception("API返回空响应")
        else:
            raise Exception("API返回无效响应")
