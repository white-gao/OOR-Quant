"""
LLM客户端基类定义
提供统一的接口规范，包含重试机制和批量处理
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseLLMClient(ABC):
    """
    LLM客户端基类，定义统一的接口

    所有具体的LLM客户端（Gemini, DeepSeek等）都应继承此类
    提供统一的重试机制和批量处理能力
    """

    def __init__(self, **config):
        """
        初始化客户端

        Args:
            **config: 模型特定的配置参数
        """
        self.config = config
        # 从配置中提取重试参数
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self._setup()

    @abstractmethod
    def _setup(self):
        """
        设置客户端（子类实现具体的初始化逻辑）
        """
        pass

    @abstractmethod
    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        调用API生成文本（子类实现具体的API调用逻辑）

        Args:
            prompt: 输入提示词
            temperature: 温度参数（控制随机性）
            max_tokens: 最大生成token数
            **kwargs: 其他模型特定参数

        Returns:
            str: 生成的文本内容

        Raises:
            Exception: API调用失败时抛出异常
        """
        pass

    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        判断错误是否可重试

        Args:
            error_msg: 错误信息

        Returns:
            bool: 是否可重试
        """
        retryable_keywords = [
            '503', '429', '500', 'timeout', 'timed out', 'deadline',
            'unavailable', 'failed to connect', 'connection',
            'rate limit', 'overload'
        ]
        return any(keyword in error_msg.lower() for keyword in retryable_keywords)

    def _generate_with_retry(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        带重试机制的生成方法（模板方法）

        Args:
            prompt: 输入提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            **kwargs: 其他参数

        Returns:
            str: 生成的文本内容

        Raises:
            Exception: API调用失败时抛出异常
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt不能为空")

        last_error = None

        # 指数退避重试机制
        for attempt in range(self.max_retries):
            try:
                # 添加请求延迟（避免限流）
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, delay * 0.3)
                    time.sleep(delay + jitter)

                # 调用子类的API实现
                return self._call_api(prompt, temperature, max_tokens, **kwargs)

            except Exception as e:
                last_error = e
                error_msg = str(e)

                # 检查是否可重试
                is_retryable = self._is_retryable_error(error_msg)

                # 如果是最后一次尝试或不可重试，直接抛出
                if attempt == self.max_retries - 1 or not is_retryable:
                    raise Exception(f"{self.__class__.__name__} API调用失败: {error_msg}")

                # 否则继续重试
                print(f"{self.__class__.__name__} API调用失败 "
                      f"(尝试 {attempt + 1}/{self.max_retries}), "
                      f"将在 {self.retry_delay}秒后重试: {error_msg[:100]}")

        # 理论上不会到这里，但为了安全
        raise Exception(f"达到最大重试次数 ({self.max_retries}): {last_error}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本内容（公开接口）

        Args:
            prompt: 输入提示词
            temperature: 温度参数（控制随机性）
            max_tokens: 最大生成token数
            **kwargs: 其他模型特定参数

        Returns:
            str: 生成的文本内容

        Raises:
            ValueError: 参数错误
            Exception: API调用失败
        """
        return self._generate_with_retry(prompt, temperature, max_tokens, **kwargs)

    def batch_generate(
        self,
        prompts: List[str],
        max_workers: int = 5,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量生成文本（支持并发）

        Args:
            prompts: 提示词列表
            max_workers: 最大并发线程数，默认5
            show_progress: 是否显示进度条，默认True
            **kwargs: 传递给generate的其他参数

        Returns:
            List[Dict]: 结果列表，每个元素包含:
                - prompt: 原始提示词
                - result: 生成的文本（成功时）
                - error: 错误信息（失败时）
                - success: 是否成功
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            if show_progress:
                print("警告: 未安装tqdm，无法显示进度条")

        def process_prompt(prompt: str, index: int) -> Dict[str, Any]:
            """处理单个提示词"""
            try:
                result = self.generate(prompt, **kwargs)
                return {
                    "index": index,
                    "prompt": prompt,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                return {
                    "index": index,
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                }

        # 并发执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_prompt, prompt, i): i
                for i, prompt in enumerate(prompts)
            }

            # 收集结果
            if show_progress and has_tqdm:
                progress = tqdm(
                    as_completed(future_to_index),
                    total=len(prompts),
                    desc=f"生成中 ({self.__class__.__name__})"
                )
            else:
                progress = as_completed(future_to_index)

            temp_results = []
            for future in progress:
                try:
                    result = future.result()
                    temp_results.append(result)
                except Exception as e:
                    index = future_to_index[future]
                    temp_results.append({
                        "index": index,
                        "prompt": prompts[index],
                        "error": f"任务执行失败: {str(e)}",
                        "success": False
                    })

        # 按原始顺序排序
        results = sorted(temp_results, key=lambda x: x["index"])

        # 移除索引字段
        for r in results:
            r.pop("index", None)

        return results

    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(config={self.config})"
