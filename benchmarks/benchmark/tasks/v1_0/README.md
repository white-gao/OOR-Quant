# OneRec Benchmark v1.0 任务模块

本目录包含 OneRec Benchmark v1.0 版本的所有任务模块。

## 架构概述

新的模块化架构将每个数据集组织为独立的模块，包含数据加载和评估逻辑。

### 核心组件

- **BaseLoader** (`base_loader.py`): 所有数据加载器的基类
- **BaseEval** (`base_evaluator.py`): 所有评估器的基类
- **TaskRegistry** (`registry.py`): 统一的任务注册表，整合了 Loader 和 Evaluator 的工厂函数

### 数据集模块结构

每个数据集模块包含：

```
dataset_name/
├── __init__.py         # 模块导出
├── config.py           # 任务配置和 prompt 模板
├── loader.py           # 数据加载器 (继承 BaseLoader)
├── evaluator.py        # 评估器 (继承 BaseEval)
└── utils.py            # 工具函数 (答案解析、指标计算等)
```


## 快速开始

### 使用模块化 Loader

```python
from benchmark.tasks.v1_0.registry import get_loader

# 获取 math_500 的 loader
loader = get_loader(
    task_name="math_500",
    benchmark_version="v1.0",
    data_dir="./data",
    model_path="your_model_path"
)

# 加载数据
data = loader.load_data(split="test")
```

### 使用模块化 Evaluator

```python
from benchmark.tasks.v1_0.registry import get_evaluator

# 获取 math_500 的 evaluator 类
evaluator_class = get_evaluator(task_name="math_500")

# 创建 evaluator 实例
evaluator = evaluator_class(
    samples=samples,  # 从 test_generated.json 读取
    task_name="math_500",
    predictions_dir="./results/model/math_500",
    debug=True
)

# 执行评估
metrics, per_sample_metrics = evaluator.evaluate()
```

## 迁移指南

详细的迁移步骤请参考 [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)。

### 迁移步骤概览

1. 创建数据集目录
2. 创建 `config.py` (任务配置)
3. 创建 `loader.py` (继承 `BaseLoader`)
4. 创建 `evaluator.py` (继承 `BaseEval`)
5. 创建 `utils.py` (工具函数)
6. 创建 `__init__.py` (模块导出)
7. 注册到 `registry.py` (统一注册表)

## 设计原则

1. **模块化**: 每个数据集独立管理自己的代码
2. **可扩展**: 通过继承基类轻松添加新数据集
3. **统一注册**: 所有任务在 `registry.py` 一处注册，避免重复定义
4. **职责分离**: 
   - **GenerationRunner**: 运行 Generator 生成结果
   - **Loader**: 负责数据加载
   - **Evaluator**: 负责评估指标计算
5. **灵活配置**: 每个数据集可以自定义配置和逻辑

## 目录结构

```
benchmark/tasks/v1_0/
├── README.md                   # 本文件
├── MIGRATION_GUIDE.md          # 迁移指南
├── __init__.py                 # 模块导出
├── base_loader.py              # Loader 基类
├── base_evaluator.py           # Evaluator 基类
├── registry.py                 # 统一任务注册表 (整合了 loader 和 evaluator 工厂)
├── label_pred/                 # label_pred 数据集模块 ✅
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── evaluator.py
│   └── utils.py
├── recommendation/             # 推荐任务模块 ✅
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── evaluator.py
│   └── utils.py
├── rec_reason/                # rec_reason 数据集模块 ✅
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── evaluator.py
│   └── utils.py
├── item_understand/                # item_understand 数据集模块 ✅
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── evaluator.py
│   └── utils.py
```

## 贡献

欢迎贡献新的数据集模块！请遵循以下步骤：

1. 参考 `math_500/` 的实现
2. 阅读 [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)
3. 创建新的数据集模块
4. 提交 Pull Request

## 许可证

与 OneRec Benchmark 主项目相同。
