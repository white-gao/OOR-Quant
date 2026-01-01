# OpenOneRec 预训练模块

OpenOneRec 预训练模块基于 Qwen3 架构，支持两阶段预训练（Itemic-Text Alignment → Full-parameter Co-Pretraining）和SFT的训练流程。

> **⚠️ 重要提示**
> 
> 本模块的分布式训练**依赖 MPI（Message Passing Interface）**进行多节点通信。当前训练脚本使用 `mpirun` 启动分布式训练，需要配置正确的 MPI 环境（如 OpenMPI）和 hostfile。
> 
> 为了简化环境配置和提升可复现性，我们计划在后续版本中发布：
> - **预配置的 Docker/Apptainer 镜像**：包含所有必要的依赖和 MPI 环境
> - **基于 torchrun 的训练脚本**：提供更易用的分布式训练启动方式
> 
> 在镜像和 torchrun 版本发布之前，请确保您的环境已正确安装和配置 MPI。


## 快速开始

### 前置要求

- **硬件**：支持 CUDA 的 GPU（建议多卡或多节点）
- **软件**：
  - Python 3.8+
  - PyTorch（支持 FSDP 和分布式训练）
  - OpenMPI 或兼容的 MPI 实现
  - NCCL（用于 GPU 间通信）
- **数据**：已转换为 Parquet 格式的训练数据（参考 `../data/README.md`）
- **模型**：Qwen3 基础模型（HuggingFace 格式）

### 1. 环境配置

首先配置训练环境：

```bash
# 设置环境变量
source set_env.sh
```

该脚本会设置必要的环境变量，包括 Python 路径、CUDA 路径等。

### 2. Qwen3 模型词表扩展

在开始训练之前，需要先对 Qwen3 基础模型进行词表扩展，以支持推荐系统特有的 item ID 编码（itemic tokens）。

#### 2.1 配置参数

编辑 `scripts/expand_qwen3_vocab.sh`，设置以下参数：

```bash
HF_MODEL_DIR=/path/to/Qwen3-0.6B          # 原始 Qwen3 HuggingFace 模型路径
OUTPUT_MODEL_DIR=/path/to/Qwen3-0.6B_itemic  # 输出扩展词表后的模型路径
ITEMIC_LAYER_N=3                          # Itemic token 的层数
VOCAB_SIZE_PER_LAYER=8192                 # 每层扩展的词表大小
```

#### 2.2 执行扩展

```bash
bash scripts/expand_qwen3_vocab.sh
```

该脚本会：
- 在原始词表基础上添加新的 itemic tokens
- 将词表大小对齐到 256 的倍数
- 初始化新 token 的 embedding 权重
- 保存扩展后的模型到指定目录

**注意**：扩展后的模型路径需要在后续训练的数据配置文件中使用（`base_model_dir` 字段）。

### 3. 数据准备

训练数据需要转换为 Parquet 格式，具体格式规范请参考 `../data/README.md`。

数据配置通过 JSON 文件指定，位于 `examples/dataset_config/` 目录下。

#### 数据配置格式

每个数据配置文件包含以下主要字段：

```json
{
    "name": "chat_completion_parquet",
    "sources": "/path/to/data_list.json",
    "base_model_dir": "/path/to/Qwen3-1.7B_itemic",
    "max_length": 30000,
    "num_epochs": 3,
    "num_workers": 2,
    "itemic_id_range": [151669, 176246],
    "add_think_pattern": false,
    "local_shuffle_buffer_size": 100000
    ...
}
```

data目录中的处理脚本会自动产出该配置文件。

### 4. 训练

训练脚本位于 `examples/` 目录下，数据配置文件位于 `examples/dataset_config/` 目录。

#### 4.1 Stage1 预训练

Stage1 主要用于训练 itemic embedding，通常需要冻结 LLM 参数，只优化 embedding 层。

```bash
# 编辑 examples/pretrain_stg1.sh，设置模型路径、输出路径等参数
bash examples/pretrain_stg1.sh
```

主要训练参数（在 `pretrain_stg1.sh` 中配置）：
- `--dataset_config examples/dataset_config/stg1.json`：指定数据配置
- `--freeze_llm`：冻结 LLM 参数
- `--start_optimize_embedding_index 151669`：从指定 token ID 开始优化 embedding
- `--model_dir`：扩展词表后的基础模型路径
- `--output_dir`：模型输出路径

#### 4.2 Stage2 预训练

Stage2 用于全参数预训练，进一步优化模型性能。该阶段会解冻所有模型参数，在推荐数据和通用文本数据的混合域上进行协同预训练。

```bash
# 编辑 examples/pretrain_stg2.sh，设置模型路径、输出路径等参数
# MODEL_DIR 应指向 Stage1 训练输出的转换后的模型路径
bash examples/pretrain_stg2.sh
```

主要训练参数（在 `pretrain_stg2.sh` 中配置）：
- `--dataset_config examples/dataset_config/pretrain.json`：指定数据配置（包含推荐数据和通用文本数据）
- `--model_dir`：Stage1 输出的转换后的模型路径
- `--output_dir`：模型输出路径
- 注意：**不包含** `--freeze_llm` 参数，表示全参数训练

#### 4.3 SFT 微调

SFT（Supervised Fine-Tuning）用于指令微调，提升模型在特定任务上的表现。该阶段在指令遵循数据上进行监督学习，使模型能够更好地理解和执行推荐相关的指令。

```bash
# 编辑 examples/posttrain_sft.sh，设置模型路径、输出路径等参数
# MODEL_DIR 应指向 Stage2 训练输出的转换后的模型路径
bash examples/posttrain_sft.sh
```

主要训练参数（在 `posttrain_sft.sh` 中配置）：
- `--dataset_config examples/dataset_config/sft.json`：指定 SFT 数据配置
- `--model_dir`：Stage2 输出的转换后的模型路径
- `--output_dir`：模型输出路径
- 数据配置中 `add_think_pattern: true` 表示数据格式会启用thinking模式，即自动增加 <think> </think>tag，以及/think和/no_think指令（用于推理任务）

## 训练配置说明

### 数据配置字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 数据加载器名称，默认为 `"chat_completion_parquet"` |
| `sources` | str | 数据文件列表路径（JSON 文件）或目录路径列表 |
| `base_model_dir` | str | 基础模型路径（扩展词表后的模型），依赖该文件对数据进行tokenizer |
| `max_length` | int | 最大序列长度 |
| `num_epochs` | int | 训练轮数 |
| `num_workers` | int | dataloader 的 worker 数量 |
| `model_class` | str | 模型类名，默认为 `"Qwen3ForCausalLM"` |
| `itemic_id_range` | list | Itemic token 的 ID 范围 `[start, end]`，仅用来进行指标统计 |
| `only_assistant_loss` | bool | 是否只计算 assistant response的损失，对chat格式数据生效 |
| `local_shuffle_buffer_size` | int | 本地 sample level 的 shuffle 缓冲区大小 |
| `add_think_pattern` | bool | 是否添加think tag （prompt中增加/think /no_think，以及在response中增加<think> </think> |

注：
* 默认dataset基于torch.utils.data.IterableDataset实现
* 默认一个GPU绑定一个进程，每个进程创建`num_workers`个worker，数据集按照总worker数量，按照文件粒度分发`sources`中的文件到各个worker，分发前会对文件list进行shuffle，读取数据时也会按照`local_shuffle_buffer_size`进行样本级shuffle
* 若`num_epochs` > 1，则进行两次训练文件分发，每次分发都会对文件list重新shuffle


### 训练参数

主要训练参数通过命令行传入 `recipes/train_qwen3.py`：

| 参数 | 说明 |
|------|------|
| `--model_dir` | 基础模型路径（huggingface格式） |
| `--output_dir` | 模型输出路径 |
| `--dataset_config` | 数据配置文件路径 |
| `--freeze_llm` | 是否冻结 LLM 参数 |
| `--learning_rate` | 学习率 |
| `--max_length` | 单step序列长度 ｜
| `--min_lr` | 最小学习率 |
| `--lr_scheduler_type` | 学习率调度器类型（如 `cosine`） |
| `--num_training_steps` | 训练步数 |
| `--save_checkpoint_per_step` | 每 N 步保存一次 checkpoint |
| `--minibatch_size` | LLM head 切分 chunk 的大小，用于分块计算 loss 以节省显存 |
| `--resume_from` | 恢复训练的 checkpoint 目录路径 |
| `--resume_from_tag` | 恢复训练的 checkpoint tag（如`global_step1000`） |
| `--resume_training_state` | 是否恢复完整的训练状态（包括优化器、学习率调度器和数据加载器状态） |
| `--start_optimize_embedding_index` | 从指定的 token ID 开始优化 embedding（用于 Stage1 训练，通常设置为 itemic tokens 的起始 ID，如 151669） |
| `--use_tie_weights` | 绑定 embedding 和 lm_head 的权重（对于 0.6B / 1.7B / 4B 等较小模型必需，以对齐 Qwen3 模型配置） |

注：
* `resume_from`用来加载框架产出的ckpt，当配置了`resume_from`优先加载该参数，仅加载`model_dir`中模型结构等参数用来初始化模型，没配置将同时加载`model_dir`中参数
* `num_training_steps` 只影响lr decay的step，该配置保证模型训练到`num_training_steps`时，lr decay到最小，但训练不会停止。建议根据token数和`max_length`配置计算最大训练step进行配置
* `max_length` 表示单step中，单卡最长的序列长度，框架会根据这个配置进行packing

## 工具脚本

### 模型转换

将训练好的 checkpoint 转换为 HuggingFace 格式：

```bash
bash scripts/convert_checkpoint_to_hf.sh <base_model_dir> <model_home> <step>
```

参数说明：
- `base_model_dir`：扩展词表后的 Qwen 基础模型目录（词表扩展阶段的输出）
- `model_home`：训练输出目录（即训练脚本中的 `OUTPUT_DIR`）
- `step`：要转换的 checkpoint 步数

**示例：**
```bash
# 假设词表扩展后的模型在 ./qwen_extended
# 训练输出在 ./output
# 要转换第 4000 步的 checkpoint
bash scripts/convert_checkpoint_to_hf.sh ./qwen_extended ./output 4000
```

转换过程：
1. 脚本会自动定位到 `{model_home}/step{step}/global_step{step}` 目录
2. 读取该目录下的训练 checkpoint
3. 将转换后的 HuggingFace 格式模型保存到 `{model_home}/step{step}/global_step{step}/converted/`

转换后的模型可以直接用于：
- HuggingFace Transformers 加载和推理
- 后续的 SFT 或其他微调阶段
- 模型评估和部署

### 模型测试

测试转换后的 HuggingFace 模型：

```bash
bash scripts/test_hf_model.sh <hf_model_dir>
```

参数说明：
- `hf_model_dir`：转换后的 HuggingFace 模型目录

**示例：**
```bash
# 测试上面转换的第 4000 步模型
bash scripts/test_hf_model.sh ./output/step4000/global_step4000/converted/
```

该脚本会验证：
- 模型权重是否正确加载
- 前向传播是否正常
- 生成功能是否可用

### 训练监控

训练过程中的日志和输出：

- **标准输出/错误**：保存在 `$OUTPUT_DIR/stdout.log` 和 `$OUTPUT_DIR/stderr.log`
- **训练日志**：包含损失值、学习率、训练步数等信息
- **TensorBoard**：模型支持 TensorBoard 可视化，可通过以下命令启动 TensorBoard 查看：
  ```bash
  tensorboard --logdir=$OUTPUT_DIR
  ```
- **Checkpoint**：按配置的步数间隔保存（`--save_checkpoint_per_step`）

### 检查点管理

训练过程中会定期保存 checkpoint，目录结构如下：

```
output_dir/
├── step50/
│   └── global_step50/
│       ├── model/          # 模型权重
│       ├── optimizer/      # 优化器状态
│       └── ...
├── step100/
│   └── global_step100/
│       └── ...
└── ...
```

**恢复训练**：
如果需要从某个 checkpoint 恢复训练，可以在训练脚本中添加：
```bash
--resume_from $OUTPUT_DIR/step1000 \
--resume_from_tag global_step1000 \
--resume_training_state
```

## 注意事项


1. **MPI 环境**：
   - 训练脚本使用 `mpirun` 进行多节点分布式训练，需要安装 OpenMPI 或兼容的 MPI 实现
   - 需要配置正确的 hostfile（如 `/etc/mpi/hostfile`），格式为每行一个节点地址
   - 确保所有节点之间可以无密码 SSH 访问
   - 训练脚本会自动读取 `OMPI_COMM_WORLD_RANK`、`OMPI_COMM_WORLD_SIZE` 等环境变量

2. **数据格式**：
   - 确保训练数据符合 Parquet 格式规范，参考 `../data/README.md`
   - 建议每个 Parquet 文件包含约 1000 个样本，便于高效加载和 shuffle
   - 数据文件列表通过 JSON 文件指定，支持本地路径和 HDFS 路径

3. **词表扩展**：
   - 训练前必须先进行词表扩展，使用扩展后的模型作为 `base_model_dir`
   - 扩展后的模型路径需要在数据配置文件的 `base_model_dir` 字段中指定
   - 确保 `itemic_id_range` 与词表扩展时的配置一致

4. **模型大小**：
   - 对于 0.6B / 1.7B / 4B 较小模型，需要添加 `--use_tie_weights` 参数以对齐Qwen3模型配置
   - 不同模型大小可能需要不同的学习率和训练步数配置

## 相关文档

- [OpenOneRec 主 README](../README.md)：项目总体介绍和完整流程
- [数据格式规范](../data/README.md)：训练数据的格式要求和预处理方法