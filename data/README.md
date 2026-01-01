# 数据集说明

本目录包含 OpenOneRec 项目的数据处理脚本和数据集格式规范。

## 数据目录说明

- **general_text/**：训练中使用到的通用文本数据，包含数学、代码、推理等领域的预训练和 SFT 数据集
- **onerec_data/**：推荐场景数据及对应的处理脚本，将原始推荐数据转换为 LLM 预训练和 SFT 训练格式

### 通用文本数据 (general_text)

通用文本数据目录包含项目使用的主要通用文本数据集信息。

`pretrain.csv` 和 `sft.csv` 两个文件中列出了所有使用的 HuggingFace 数据集 URL 和对应的样本数量。为了方便大家复现，我们也在 HuggingFace 上放出了我们处理之后的数据集：

- [预训练数据集 HuggingFace 链接](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)
- [SFT数据集 HuggingFace 链接](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT)

> **NOTE**：我们给出的 HuggingFace 处理好的数据中，目前还不包含部分数据集（Nemotron_CC_Math_v1，Nemotron_Pretraining_Code_v1，Nemotron_CC_v2），后续会直接给出一个数据处理脚本，方便大家复现。

### OneRec 业务数据 (onerec_data)

OneRec 业务数据目录包含推荐系统相关的数据处理脚本，将原始数据转换为 LLM 预训练和 SFT 训练格式。包含视频推荐、用户画像、交互推荐、标签预测、跨域推荐等多种推荐场景的数据处理脚本。

- [OpenOneRec数据集 HuggingFace 链接](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)

## 数据集格式规范

为了统一处理数据，我们使用统一的 Parquet 数据格式，每个 Parquet 文件包含以下字段：

### 字段说明

| 字段名 | 类型 | 必需 | 默认值 | 描述 | 要求 |
|--------|------|------|--------|------|------|
| uuid | str | 是 | 自动生成UUID | 唯一标识符 | 必须是有效的UUID格式，同一个数据集内必须要唯一 |
| source | str | 是 | - | 数据来源标识 | 不能为空字符串 |
| metadata | str | 否 | "{}" | JSON格式的元数据字典 | 必须是有效的JSON字典字符串 |
| images | str | 否 | "{}" | （废弃）本项目仅训练文本，此字段不使用 | - |
| videos | str | 否 | "{}" | （废弃）本项目仅训练文本，此字段不使用 | - |
| messages | str | 否 | None | JSON格式的消息列表，用于对话格式数据 | 必须是有效的JSON数组，每个消息必须有role和content字段 |
| segments | str | 否 | None | JSON格式的段落列表，用于分段数据 | 必须是有效的JSON数组，每个段落必须有type字段 |
| image | str | 否 | None | （废弃）本项目仅训练文本，此字段不使用 | - |
| video | str | 否 | None | （废弃）本项目仅训练文本，此字段不使用 | - |
| text | str | 否 | None | 文本内容 | 无特殊要求 |
| label | str | 否 | None | 标签信息，如果`image`,`video`,`text`存在，则为对应的label | 无特殊要求 |

### 数据格式示例

数据格式支持两种主要类型：
- **Segments格式**：用于普通文本数据，使用 `segments` 字段存储文本段落列表
- **Chat格式**：用于对话数据，使用 `messages` 字段存储对话消息列表

**Chat格式数据（对话数据）：**

| 字段 | 值 |
|------|-----|
| uuid | 550e8400-e29b-41d4-a716-446655440001 |
| source | conversation_dataset |
| metadata | '{}' |
| images | '{}' |
| videos | '{}' |
| messages | '[{"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Machine learning is a subset of artificial intelligence."}]}]' |

**Segments格式数据（普通文本）：**

| 字段 | 值 |
|------|-----|
| uuid | 550e8400-e29b-41d4-a716-446655440002 |
| source | document_dataset |
| metadata | '{}' |
| images | '{}' |
| videos | '{}' |
| segments | '[{"type": "text", "text": "Introduction paragraph..."}, {"type": "text", "text": "Main content..."}]' |

### 字段验证规则

| 验证项 | 规则说明 |
|--------|----------|
| JSON字段验证 | metadata必须是有效的JSON字典字符串；images和videos字段（废弃）应设置为"{}" |
| 消息格式验证 | messages字段（如果存在）必须包含有效的消息列表，每个消息必须有role和content字段 |
| 角色验证 | 消息的role必须是user、assistant或system之一 |
| 内容类型验证 | 消息内容中的type必须是text（本项目仅训练文本，不支持image和video类型） |
| 段落格式验证 | segments字段（如果存在）必须包含有效的段落列表，每个段落必须有type字段，type应为"text" |

### 文件大小建议

为了便于 DataLoader 高效加载数据，建议每个 Parquet 文件包含约 **1000 个样本**。如果数据量较大，可以使用分片（sharding）的方式将数据分割成多个文件，文件名格式建议为：

```
part-00000-of-00010.parquet
part-00001-of-00010.parquet
...
part-00009-of-00010.parquet
```

## 快速开始

### 1. 下载数据集

首先需要到 HuggingFace 下载对应数据集：

- [预训练通用文本数据集](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)
- [SFT通用文本数据集](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT)
- [OneRec推荐数据集](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)

### 2. 处理推荐数据

编辑 `onerec_data/run.sh` 设置以下路径：

```bash
INPUT_METADATA="path/to/onerec_bench_release.parquet"
PID2SID_MAPPING="path/to/video_ad_pid2sid.parquet"
PRODUCT_PID2SID_MAPPING="path/to/product_pid2sid.parquet"
CAPTION_INPUT="path/to/pid2caption.parquet"
OUTPUT_BASE_DIR="./output"
```

然后运行：

```bash
cd data/onerec_data
bash run.sh
```

### 3. 预训练数据分片处理

生成的数据可以分别调用 prepare 脚本来处理。编辑 `prepare_pretrain.sh` 或 `prepare_sft.sh`，修改以下配置：

```bash
GENERAL_TEXT_PATH="data/general_text"      # 通用文本数据路径
REC_DATA_PATH="data/onerec_data/output"   # 推荐数据输出路径
OUTPUT_DIR="./output/split_data"          # 最终输出路径
MAX_ROWS=1000                             # 每个文件包含的样本数
```

然后运行：

```bash
# 处理预训练数据
bash prepare_pretrain.sh

# 处理 SFT 数据
bash prepare_sft.sh
```

### 4. Distillation 数据处理

用于 on-policy distillation 的数据处理。编辑 `prepare_distillation.sh`，修改以下配置：

```bash
INPUT_PATH="data/general_text"                    # 通用文本数据路径
OUTPUT_FILE="./output/onpolicy_distillation.parquet"  # 输出文件路径
NUM_SAMPLES=200000                                # 采样样本数量
SEED=42                                           # 随机种子
```

然后运行：

```bash
bash prepare_distillation.sh
```

### 5. RL 数据处理

用于强化学习（RL）训练的数据处理。将多个 RL 任务的数据集合并后切分为训练集和测试集。编辑 `prepare_rl.sh`，修改以下配置：

```bash
REC_DATA_PATH="data/onerec_data"                  # onerec 数据集路径
OUTPUT_DIR="./output/rl_data"                     # 输出目录路径
TEST_SIZE=1000                                     # 每个子任务测试集样本数量
SEED=42                                            # 随机种子
```

脚本会处理以下 5 个 RL 任务的数据集：
- `sft_video_rec.parquet` - 视频推荐任务
- `sft_ad_rec.parquet` - 广告推荐任务
- `sft_product_rec.parquet` - 商品推荐任务
- `sft_interactive_rec.parquet` - 交互推荐任务
- `sft_label_cond_rec.parquet` - label条件推荐任务

然后运行：

```bash
bash prepare_rl.sh
```

输出结果：
- `./output/rl_data/train.parquet` - 训练集（所有任务合并后的剩余数据）
- `./output/rl_data/test.parquet` - 测试集（从合并数据中随机抽取的 1000 条样本）


## 注意事项

* 所有脚本默认只处理 `split=0`（训练集）数据