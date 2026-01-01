# LLM API 统一封装

这是一个统一的LLM API封装库，提供了简洁、优雅的接口来调用不同的大语言模型。

## 特性

✅ **统一接口** - 所有模型使用相同的调用方式
✅ **易于切换** - 更换模型只需修改一个参数
✅ **配置管理** - 支持JSON配置文件和直接传参
✅ **并发处理** - 内置批量生成和并发支持
✅ **错误处理** - 自动重试机制，提高调用成功率
✅ **进度显示** - 批量处理时显示进度条

## 支持的模型

- **Claude** - Anthropic Claude模型
- **Gemini** - Google Vertex AI Gemini模型
- **DeepSeek** - 通过百度千帆平台调用DeepSeek模型

## 模型价格对比 (2025年11月)

可能有幻觉

### Claude (Anthropic)

| 模型 | Model ID | 输入 ($/1M tokens) | 输出 ($/1M tokens) |
|------|----------|-------------------|-------------------|
| Claude Opus 4.1 | `claude-opus-4-20250514` | $15.00 | $75.00 |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | $3.00 (≤200K) / $6.00 (>200K) | $15.00 (≤200K) / $22.50 (>200K) |
| Claude Sonnet 4 | `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| Claude Haiku 4.5 | `claude-haiku-4-5-20250929` | $1.00 | $5.00 |
| Claude Haiku 3.5 | `claude-3-5-haiku-20241022` | $0.80 | $4.00 |

### Gemini (Google)

| 模型 | Model ID | 输入 ($/1M tokens) | 输出 ($/1M tokens) |
|------|----------|-------------------|-------------------|
| Gemini 3 Pro | `gemini-3-pro-preview` | $2.00 (≤200K) / $4.00 (>200K) | $12.00 (≤200K) / $18.00 (>200K) |
| Gemini 2.5 Pro | `gemini-2.5-pro` | $1.25 (≤200K) / $2.50 (>200K) | $10.00 (≤200K) / $15.00 (>200K) |
| Gemini 2.5 Flash | `gemini-2.5-flash` | $0.30 | $2.50 |
| Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` | $0.10 | $0.40 |

### DeepSeek

| 模型 | Model ID | 输入 ($/1M tokens) | 输出 ($/1M tokens) |
|------|----------|-------------------|-------------------|
| DeepSeek-V3.2-Exp (非思考) | `deepseek-chat` | $0.028 (cache hit) / $0.28 (miss) | $0.42 |
| DeepSeek-V3.2-Exp (思考) | `deepseek-reasoner` | $0.028 (cache hit) / $0.28 (miss) | $0.42 |

> **注意**: 价格可能会有变动，请以官方最新定价为准。
> - Claude: https://claude.com/pricing
> - Gemini: https://ai.google.dev/gemini-api/docs/pricing
> - DeepSeek: https://api-docs.deepseek.com/quick_start/pricing

## 安装依赖

```bash
pip install openai google-cloud-aiplatform anthropic tqdm
```

## 快速开始

### 1. 配置文件方式（推荐）

首先，编辑 `api/config/llm_config.json` 填入您的配置：

```json
{
  "claude": {
    "api_key": "your-anthropic-api-key",
    "base_url": "https://api.anthropic.com",
    "model_name": "claude-sonnet-4-20250514"
  },
  "gemini": {
    "project": "your-gcp-project-id",
    "location": "us-central1",
    "model_name": "gemini-2.5-pro",
    "credentials_path": "path/to/credentials.json"
  },
  "deepseek": {
    "api_key": "your-api-key",
    "base_url": "https://qianfan.baidubce.com/v2",
    "model": "deepseek-r1",
    "appid": "your-appid"
  }
}
```

然后使用：

```python
from api import get_client_from_config

# 创建客户端
client = get_client_from_config("gemini")

# 生成文本
response = client.generate("讲个笑话")
print(response)
```

### 2. 直接传参方式

```python
from api import get_client

# Gemini
client = get_client(
    "gemini",
    project="your-project",
    location="us-central1",
    model_name="gemini-2.5-pro",
    credentials_path="path/to/credentials.json"
)

# DeepSeek
client = get_client(
    "deepseek",
    api_key="your-api-key",
    appid="your-appid"
)

response = client.generate("你好，介绍一下你自己")
print(response)
```

### 3. 批量生成

```python
from api import batch_generate

# 准备多个提示词
prompts = [
    "什么是机器学习？",
    "解释一下深度学习",
    "神经网络的原理是什么？"
]

# 批量生成（自动并发处理）
results = batch_generate(
    prompts=prompts,
    model="gemini",
    max_workers=3,  # 并发线程数
    show_progress=True  # 显示进度条
)

# 处理结果
for item in results:
    if item["success"]:
        print(f"Q: {item['prompt']}")
        print(f"A: {item['result']}\n")
    else:
        print(f"失败: {item['prompt']}")
        print(f"错误: {item['error']}\n")
```

## API文档

### get_client(model, **config)

创建LLM客户端实例。

**参数：**
- `model` (str): 模型名称 ("claude", "gemini" 或 "deepseek")
- `**config`: 模型特定的配置参数

**返回：** `BaseLLMClient` 实例

**Claude配置参数：**
- `api_key` (str, 必需): Anthropic API密钥
- `base_url` (str, 可选): API基础URL，默认"https://api.anthropic.com"
- `model_name` (str, 可选): 模型名称，默认"claude-sonnet-4-20250514"
- `max_retries` (int, 可选): 最大重试次数，默认3
- `retry_delay` (int, 可选): 重试延迟秒数，默认2

**Gemini配置参数：**
- `project` (str, 必需): GCP项目ID
- `location` (str, 必需): 区域
- `model_name` (str, 可选): 模型名称，默认"gemini-2.5-pro"
- `credentials_path` (str, 可选): 凭证文件路径
- `max_retries` (int, 可选): 最大重试次数，默认3
- `retry_delay` (int, 可选): 重试延迟秒数，默认2

**DeepSeek配置参数：**
- `api_key` (str, 必需): API密钥
- `appid` (str, 必需): 应用ID
- `base_url` (str, 可选): API基础URL，默认百度千帆
- `model` (str, 可选): 模型名称，默认"deepseek-r1"
- `max_retries` (int, 可选): 最大重试次数，默认3
- `retry_delay` (int, 可选): 重试延迟秒数，默认2

### client.generate(prompt, temperature=None, max_tokens=None, **kwargs)

生成文本内容。

**参数：**
- `prompt` (str): 输入提示词
- `temperature` (float, 可选): 温度参数，控制随机性
- `max_tokens` (int, 可选): 最大生成token数
- `**kwargs`: 其他模型特定参数

**返回：** str - 生成的文本

**异常：**
- `ValueError`: 参数错误
- `Exception`: API调用失败

### get_client_from_config(model, config_path=None)

从配置文件创建客户端。

**参数：**
- `model` (str): 模型名称
- `config_path` (str, 可选): 配置文件路径

**返回：** `BaseLLMClient` 实例

### batch_generate(prompts, model, max_workers=5, show_progress=True, config_path=None, **config)

批量生成文本。

**参数：**
- `prompts` (List[str]): 提示词列表
- `model` (str): 模型名称
- `max_workers` (int, 可选): 最大并发数，默认5
- `show_progress` (bool, 可选): 是否显示进度条，默认True
- `config_path` (str, 可选): 配置文件路径
- `**config`: 模型配置参数

**返回：** List[Dict] - 结果列表，每个元素包含：
- `prompt` (str): 原始提示词
- `result` (str): 生成的文本（成功时）
- `error` (str): 错误信息（失败时）
- `success` (bool): 是否成功

## 项目结构

```
api/
├── __init__.py          # 主入口，工厂函数和工具函数
├── base.py              # 基类定义
├── claude.py            # Claude客户端实现
├── gemini.py            # Gemini客户端实现
├── deepseek.py          # DeepSeek客户端实现
├── config/
│   ├── llm_config.json  # 配置文件
│   └── README.md        # 配置说明
└── README.md            # 本文件
```

## 高级用法

### 自定义参数

```python
# 控制生成参数
response = client.generate(
    "写一首诗",
    temperature=0.8,      # 提高创造性
    max_tokens=500        # 限制输出长度
)
```

### 错误处理

```python
try:
    response = client.generate("你的问题")
    print(response)
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"API调用失败: {e}")
```

### 切换模型

```python
# 相同的代码，只需修改模型名称
for model_name in ["claude", "gemini", "deepseek"]:
    client = get_client_from_config(model_name)
    response = client.generate("你好")
    print(f"{model_name}: {response}")
```

## 注意事项

1. **配置文件安全**：不要将包含真实密钥的配置文件提交到版本控制
2. **并发限制**：注意API的并发限制，避免触发限流
3. **重试机制**：默认会自动重试3次，遇到临时错误会自动处理
4. **依赖版本**：确保安装了兼容的依赖包版本

## 许可证

MIT License
