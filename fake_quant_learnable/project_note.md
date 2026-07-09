# fake_quant_learnable Project Note

更新时间：2026-06-04

`fake_quant_learnable` 是当前 OneRec 推荐大模型 PTQ 量化实验的轻量主路径。当前 active 代码只保留 naive W8A8 fake quant 和 SmoothQuant W8A8；LWC/LET、tail-protect 等历史实验实现已经归档到 `support/archived_lwc_let/`，不再作为主入口维护。

这份文档不是完整论文，也不是最终 README；它的目标是让后续读代码的人快速知道：当前在做什么、代码怎么分布、已有结果说明了什么、下一步应该往哪里改。

## 研究目标

当前任务是 OneRec 的 AD 推荐预测。模型输入是用户历史和指令 prompt，输出是 SID 序列，评测通过 beam search 生成 top-k SID，再映射到 item/PID 后计算 pass@k、recall@k、pid_pass@k、pid_recall@k。

量化目标主要是 W8A8：

- weight：per-output-channel absmax scale，`torch.float8_e4m3fn` fake quant / dequant。
- activation：默认 per-token absmax scale，`torch.float8_e4m3fn` fake quant / dequant。
- activation sharing：对实际共享输入的位置统一 QDQ，例如 attention 的 q/k/v 输入、FFN 的 gate/up 输入。
- generation：默认 HuggingFace `generate`，`num_beams=32`，`num_return_sequences=32`，`max_new_tokens=3`。

研究最初假设是：通过校准集优化量化参数，降低 block output MSE，可以减小量化损失并提升推荐指标。但目前实验显示，常规平滑/重构类 PTQ 在 OneRec AD 上的收益非常有限，需要进一步转向更推荐任务相关的目标或保护策略。

## 当前方法

### baseline_w8a8

基础 W8A8 fake quant。逐层把 `nn.Linear` 替换为 `BaselineFakeQuantLinear`，权重量化采用 per-output-channel FP8 QDQ，激活默认采用 per-token FP8 QDQ。

这是当前所有后续推荐专用 PTQ 实验的主要对照组。

### smoothquant_w8a8

先用 calibration prompt 收集每个 Linear 输入激活 absmax，再按 SmoothQuant 公式计算 scale，对 q/k/v、o_proj、gate/up/down 等 OmniQuant 风格位置做平滑。当前实现支持 fold：能等价折叠的位置会把 scale 合并进 LayerNorm 或相邻 Linear，量化时输入分布和权重分布更接近实际部署设置。

当前 active 代码中，SmoothQuant 使用固定 scale 的 `SmoothQuantFakeQuantLinear`，没有任何可学习参数。已有结果显示 SmoothQuant 确实能显著改变 activation/weight 数值分布，但最终推荐指标提升不稳定、幅度很小。

### 已归档：m1_lwt / m2_let / m2_lwt_let

M1/M2 曾经用于验证 OmniQuant 风格的 learnable clipping 和 learnable equivalent transform：

- `m1_lwt`：学习 weight clipping threshold；
- `m2_let`：只训练 LET scale；
- `m2_lwt_let`：同时训练 LET 和 LWT，可用 SmoothQuant scale 初始化 LET。

这些实现已经从 active 主代码剥离，保存在 `fake_quant_learnable/support/archived_lwc_let/`。剥离原因是：大量实验显示 LWC+LET 的 block MSE 下降幅度很小，推荐指标也没有稳定优于 naive W8A8 / SmoothQuant W8A8。后续如果需要回顾实现或复现实验，可以从归档目录读取，但不要从 active runner 继续调用这些路径。

## 代码结构

```text
fake_quant_learnable/
  quant.py                  # FP8 E4M3 QDQ forward-only fake quant
  modules.py                # BaselineFakeQuantLinear / SmoothQuantFakeQuantLinear
  apply.py                  # Linear 替换、shared input QDQ、SQ scope 判断
  run_m1_onerec_ad.py       # 主 runner：baseline_w8a8 / smoothquant_w8a8 生成评测
  run_learnable_quant_ad.sh # 统一 shell 入口，保留旧文件名但只跑 W8A8/SQ
  support/
    runtime_utils.py        # tensor tree detach/device helper
    smoothquant_runtime.py  # SmoothQuant scale 收集、fold、W8A8 包装
    inspect_smoothquant_distribution.py # SQ 前后 activation/weight 分布探针
    archived_lwc_let/       # 历史 LWC/LET/tail-protect 实现快照，不作为 active path
  tests/                    # W8A8/SQ toy model 单测和 runner 单测
  results/                  # 本地实验输出，不应当视为干净源码
```

主 runner 的默认配置集中在 `run_m1_onerec_ad.py`：

```text
model path: /home/guowei/OneRec-1.7B/
data dir: data/onerec_data/benchmark-data-calib1024
split: test
calib split: 如果存在 ad_calib.parquet，则使用 calib
act quant: per_token
act quant mode: shared_input
dtype: bfloat16
num_beams: 32
num_return_sequences: 32
max_new_tokens: 3
```

## 常用命令

统一入口：

```bash
bash fake_quant_learnable/run_learnable_quant_ad.sh
```

W8A8 baseline：

```bash
MODE=baseline_w8a8 DEVICE=cuda:0 RUN_NAME=baseline_w8a8_1p7b_ad_calib1024 \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

SmoothQuant W8A8：

```bash
MODE=smoothquant_w8a8 DEVICE=cuda:0 RUN_NAME=smoothquant_w8a8_1p7b_ad_calib1024 \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

SQ 分布/误差分析脚本：

```bash
python3 -m fake_quant_learnable.support.inspect_smoothquant_distribution \
  --device cuda:0 --sq_calib_sample_size 3
```

开启辅助 SID teacher-forcing NLL/PPL 指标：

```bash
COMPUTE_SID_PPL=1 MODE=baseline_w8a8 DEVICE=cuda:0 RUN_NAME=baseline_w8a8_sidppl \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

`SID_PPL_MAX_ITEMS=1` 是默认值，表示每条样本只取第一个 ground-truth SID item 计算三个 SID token 的 teacher-forcing NLL/PPL。增大这个值会线性增加额外 forward 开销。

## 当前实验结果摘要

下面结果来自当前目录下已有 `eval_results.json`。全量结果使用 calib1024 切分后的 `test`，总样本数 26653。旧 AD1000 结果只作为历史参考，不能和全量结果直接比较。

### 全量 calib1024 / test


| run                                                 |   pass@1 |  pass@16 |  pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | 备注                                               |
| ----------------------------------------------------- | ---------: | ---------: | ---------: | ----------: | ------------: | --------------: | ---------------------------------------------------- |
| baseline_w8a8_1p7b_ad_calib1024                     | 0.019773 | 0.139947 | 0.204705 |  0.070265 |    0.190748 |      0.064848 | 当前主要 W8A8 对照                                 |
| smoothquant_w8a8_1p7b_ad_calib1024_fixed_fold_full  | 0.018760 | 0.139196 | 0.202416 |  0.069607 |    0.188759 |      0.064348 | SQ fold，效果略低于 baseline                       |
| smoothquant_w8a8_fold_ad_calib1024_1p7b             | 0.019097 | 0.139309 | 0.204893 |  0.070519 |    0.190710 |      0.065231 | SQ fold，和 baseline 基本持平                      |
| smoothquant_w8a8_alpha0p4                           | 0.019510 | 0.142948 | 0.203955 |  0.070333 |    0.189322 |      0.064769 | alpha=0.4，@16 略好，@32 持平                      |
| m2_lwt_let_1p7b_ad_calib1024_sqinit_fixed_fold_full | 0.018572 | 0.140922 | 0.202491 |  0.070030 |    0.188984 |      0.064747 | LWT+LET，SQ 初始化，未优于 baseline                |
| m2_lwt_let_sqinit_ad_calib1024_1p7b                 | 0.018872 | 0.140134 | 0.202229 |  0.068842 |    0.188009 |      0.063464 | 新版 SQ init，效果偏低                             |
| baseline_w8a8_skip_last_ad_calib1024_1p7b_full      | 0.019322 | 0.140697 | 0.205118 |  0.070617 |    0.191423 |      0.065281 | 最后一层不量化，和 baseline 接近                   |
| w8a8_tail1_ad_calib1024_1p7b_full                   | 0.020185 | 0.145500 | 0.211533 |  0.072724 |    0.197351 |      0.067080 | tail token activation 保护，结果较好但需要复跑确认 |

### 梯度 group 权重 probe / 2026-06-15

目的：在实现 `grad_weighted_gptq_fp8_w8a8` 的 layer-wise group 版本前，先看 calib 梯度统计出的 group 权重和手工 group 权重的差异。

设置：OneRec-1.7B，calib 前 32 条，全 28 层；梯度目标沿用当前 v2，即 `last prompt position -> first ground-truth s_a` 的 CE loss；token sensitivity 为 `mean(|hidden * grad|)`，再使用当前 gradient config 做 clip / floor / mean normalize，最后按 prompt role group 聚合。

手工权重 raw 为 `text=10, history_sid=1, interest_sid=5, sid_boundary=2`。由于 GPTQ Hessian 收集前会按样本均值归一化，probe 中手工 group 的 normalized 均值如下：


| group        | manual normalized |
| -------------- | ------------------: |
| text         |            3.5205 |
| history_sid  |            0.3548 |
| interest_sid |            1.6937 |
| sid_boundary |            0.7018 |

逐层梯度 group 权重：


|  layer |   text | history_sid | interest_sid | sid_boundary |
| -------: | -------: | ------------: | -------------: | -------------: |
| manual | 3.5205 |      0.3548 |       1.6937 |       0.7018 |
|      0 | 2.3065 |      0.8292 |       1.9489 |       0.4468 |
|      1 | 2.7725 |      0.7827 |       1.9727 |       0.3470 |
|      2 | 2.8290 |      0.7375 |       1.9075 |       0.4044 |
|      3 | 2.9290 |      0.7132 |       1.8845 |       0.4094 |
|      4 | 2.9798 |      0.6946 |       1.8580 |       0.4244 |
|      5 | 3.1133 |      0.6678 |       1.8124 |       0.4301 |
|      6 | 3.2066 |      0.6422 |       1.7805 |       0.4419 |
|      7 | 3.2373 |      0.6258 |       1.7633 |       0.4572 |
|      8 | 3.2671 |      0.6048 |       1.7324 |       0.4829 |
|      9 | 3.3374 |      0.5702 |       1.7123 |       0.5077 |
|     10 | 3.4621 |      0.5054 |       1.6778 |       0.5551 |
|     11 | 3.4867 |      0.4855 |       1.6674 |       0.5738 |
|     12 | 3.4977 |      0.4838 |       1.6727 |       0.5704 |
|     13 | 3.3820 |      0.4832 |       1.7018 |       0.5962 |
|     14 | 3.2321 |      0.4942 |       1.7700 |       0.6053 |
|     15 | 3.1795 |      0.5023 |       1.8379 |       0.5878 |
|     16 | 3.1632 |      0.4928 |       1.8752 |       0.5900 |
|     17 | 3.1142 |      0.5194 |       1.8762 |       0.5747 |
|     18 | 3.1354 |      0.5289 |       1.9471 |       0.5317 |
|     19 | 3.1066 |      0.5393 |       1.9286 |       0.5356 |
|     20 | 2.9439 |      0.5701 |       1.9664 |       0.5371 |
|     21 | 2.9470 |      0.5453 |       1.9674 |       0.5639 |
|     22 | 3.1310 |      0.5596 |       2.1680 |       0.4183 |
|     23 | 2.6226 |      0.6035 |       2.3082 |       0.4747 |
|     24 | 2.5130 |      0.6440 |       2.4645 |       0.4060 |
|     25 | 2.3945 |      0.6588 |       2.3881 |       0.4534 |
|     26 | 2.2611 |      0.6731 |       2.2838 |       0.5162 |
|     27 | 3.2884 |      0.5272 |       1.1557 |       0.7731 |

主要观察：

1. 梯度 group 版本显著抬高 history SID：手工 normalized 为 `0.3548`，梯度大多在 `0.48-0.83`。由于 history SID token 数多，这会明显改变 Hessian 主导方向。
2. text 权重整体低于手工版，只在 layer 10-12 接近 `3.52`，后段 layer 23-26 降到 `2.26-2.62`。
3. interest SID 有明显 layer-wise 信号，中后层尤其 layer 22-26 升到 `2.17-2.46`，高于手工 `1.69`。
4. boundary 大多数层低于手工，最后一层升到 `0.7731`，说明 boundary 影响可能更偏输出侧。
5. 当前代码已将 `grad_weighted_gptq_fp8_w8a8` 改为纯梯度 layer-wise group 权重：先用 calib 梯度得到逐层 group 均值，再按 group 回填 per-token Hessian 权重；未引入手工 prior shrinkage。

## 基于MLLM量化的相关尝试

如果将当前的目标量化模型——推荐大模型当成多模态大模型，可以找到一个大概的对应关系，SID token相比大模型原本的token其实是异源token，所以一定程度上可以当成一个输入模态，从而借鉴MLLM领域量化的思路和技术。

经过调研后发现，MLLM量化和大模型量化有比较明显的不同，由于MLLM的token来源很多，有LLM原生的文本token，也有从视觉token，音频token等其他来源，虽然在训练期间会将其他模态token对齐到LLM原生的文本空间中，但是还是会产生modality gap，这个gap就成为了绝大部分MLLM量化算法的动机，在这里列举部分：

- 模态token的激活值分布和文本token存在明显不同，例如模态token的范围是文本toke的20x
- 模态token和文本token对end loss的梯度存在不同，也可以理解成不同token的影响力/敏感度不同，部分模态的token存在信息冗余现象
- 模态token和文本token的attention不同

**找proxy metric来对token进行打分**

可以看出MLLM的问题集中出现在token维度上，而传统大模型的量化算法基本都是观察到通道的异常值分布，而在channel维度做文章。token和channel两个维度是正交的，channel优化的方法不会显式地对token进行区分，从而导致传统channel-centric的量化方法在MLLM上量化效果不好。而目前MLLM主要的解决思路还是在token上做出区分，是一个在token维度上更细粒度量化的方法，包括但不限于：

- 对不同的token应用不同的量化粒度/分块处理
- 在传统的逐层优化mse的基础框架下，利用某种metric计算token的加权，得到更准确的优化目标函数，主要应用在这两大方法/技术路线
  - scale的优化
  - GPTQ的流派

基于以上这些信息，我目前想要参考MLLM的量化思路/技术，在当前的推荐模型进行量化，参考MLLM中的不同动机，做了以下初步动机探索：

- SID token的激活值：目前没有发现明显的激活值gap
- SID token的梯度：
  目前发现梯度有比较多的信息，主要包括以下几点：
  1. SID token整体的梯度要比文本token要小
  2. SID内部token也有梯度区别，历史交互和感兴趣视频的sid token也有梯度区别

但是目前的卡点是，我有了这些信息，但是我不知道如何设计算法，因为之前的SQ算法没有收益，所以下一步不知道怎么进行

## 当前主线方案总结：GPTQ token 加权 weight 量化 + dynamic activation + tail1 / 2026-06-17

当前最重要的主线组合是：`grad_weighted_gptq_fp8_w8a8_tail1`，也就是 **v3 逐层梯度 group token 加权 GPTQ weight 量化 + activation per-token dynamic fake quant + tail1 BF16 activation 保护**。这个方案的核心思想是：weight 量化仍然用 GPTQ 的逐层二阶误差补偿框架，但在构造 GPTQ Hessian 时，不把所有校准 token 等权看待，而是根据推荐 prompt 中不同 token 类型的重要性，对校准激活 `X` 做 token 加权；推理阶段再配合 shared-input 的 dynamic activation quant，并把 `<|sid_begin|>` 和 decode 阶段 activation 保留为原始 BF16。

### 1. 整体 pipeline

当前 runner 是 `fake_quant_learnable/run_m1_onerec_ad.py`。主要 mode 包括：

- `gptq_fp8_w8a8`：普通 GPTQ weight + activation per-token dynamic。
- `weighted_gptq_fp8_w8a8`：手工 token group 加权 GPTQ，不带 tail1。
- `weighted_gptq_fp8_w8a8_tail1`：手工 token group 加权 GPTQ，带 tail1。
- `grad_weighted_gptq_fp8_w8a8`：当前已经改为 v3 的逐层梯度 group token 加权 GPTQ，不带 tail1。
- `grad_weighted_gptq_fp8_w8a8_tail1`：当前最关注的组合，v3 逐层梯度 group token 加权 GPTQ，带 tail1。

默认配置里 activation 使用 `act_quant="per_token"`，activation mode 使用 `act_quant_mode="shared_input"`，GPTQ 默认 `damp_percent=0.01`、`block_size=128`。calib 默认使用 `benchmark-data-calib1024` 下对应 task 的 `*_calib.parquet`，如果不存在 calib split 就回退到 test split。每条 calib 样本会先用 `format_prompt(sample["prompt"], prompt_token)` 把任务 prompt 和 generation prompt token 拼起来；当前 item prediction 的 prompt token 通常是 `<|sid_begin|>`，因此 prefill 最后一个 token 就是即将生成 SID 的起始标记。

逐层（layer）量化时，`apply_gptq_fp8_layers` 对每个被选中的 Transformer block 做：

1. 用 hooks 捕获该层的 FP teacher 输入 `fp_inputs`。
2. 在该层内部所有 `nn.Linear` 上收集 GPTQ Hessian。
3. 用 FP teacher block 继续前传，得到下一层的 FP 输入。
4. deepcopy 当前 block，基于 Hessian 做 GPTQ weight fake quant，然后替换成 `GPTQFakeQuantLinear`。
5. 如果是 shared-input activation quant，就 patch Qwen3 attention/MLP forward，让共享输入只 quantize 一次。

注意：当前实现仍然是 teacher-block 推进，下一层 Hessian 的输入来自 FP teacher block 输出，不是来自已量化 block 的输出。这个 sequential quantized-input GPTQ 已经作为 TODO 记录过，后续可以单独做 ablation。

### 2. GPTQ weight 量化逻辑

GPTQ 的 Hessian 收集在 `fake_quant_learnable/gptq.py::collect_gptq_hessians`。对每个 Linear，hook 到它的输入 `x`，把最后一维当作 hidden/channel 维，其余 batch/seq 维 flatten 成二维矩阵：

```text
X in R^{N x d_in}
```

普通 GPTQ 使用：

```text
H = (X^T X) / N
```

token 加权版本使用：

```text
H = (X^T diag(w) X) / sum(w)
```

其中 `w` 是形状和 `x.shape[:-1]` 对齐的 token 权重。也就是说，token 加权不会改变模型 forward，也不会改变推理时的 logits；它只改变 GPTQ 校准期间看到的二阶输入分布，让 GPTQ 在做 weight 量化误差补偿时更偏向保护高权重 token 对应的输入方向。

weight 量化本身在 `gptq_fp8_quantize_weight` 中完成。当前逻辑是：

1. 对 Hessian 做对称化：`H = 0.5 * (H + H.T)`。
2. 对 dead channel 做处理：如果 Hessian diagonal 小于 eps，则把该 channel 的 Hessian diag 设为 1，并把对应 weight column 置 0。
3. 加 damp：`H_ii += damp_percent * mean(live_diag)`，默认 `damp_percent=0.01`。
4. 用 Cholesky 求 Hessian inverse factor，失败时逐步加 jitter，增强 8B 上的稳定性。
5. FP8 weight scale 是按 output channel 计算的 row-wise scale：

```text
scale_row = max(abs(W_row)) / 448
```

6. 按 GPTQ 的列顺序做 block-wise error compensation，默认 `block_size=128`。每列先吸附到 FP8 E4M3 fake-quant 网格，再把 `(w-q)/d` 的误差传播到同一 block 后续列，以及 block 之后的列。

当前仍是 fake quant：`weight_qdq` 存的是量化再反量化后的 tensor，并注册到 `GPTQFakeQuantLinear` buffer 中；它模拟真实部署中的 FP8 数值误差，但 PyTorch fake 路径本身不会吃到真实 FP8 GEMM 的吞吐收益。

### 3. token 加权方案一：手工 prompt-role group 加权

手工加权由 `fake_quant_learnable/token_weights.py::PromptTokenWeightConfig` 定义，默认 raw 权重是：


| prompt role  | raw weight |
| -------------- | -----------: |
| text         |       10.0 |
| history_sid  |        1.0 |
| interest_sid |        5.0 |
| sid_boundary |        2.0 |

prompt role 的识别方式是：

- SID item 匹配形如 `<|sid_begin|><s_a_*><s_b_*><s_c_*><|sid_end|>` 的片段。
- `<s_a_*>/<s_b_*>/<s_c_*>` 属于 SID code token。
- `<|sid_begin|>` 和 `<|sid_end|>` 属于 `sid_boundary`。
- SID item 会按 prompt 中相邻 SID item 之间是否有文字/数字/中文 section break 分组；第一组 SID 作为 `history_sid`，后续组作为 `interest_sid`。
- 其他 token 都作为 `text`。

优先使用 tokenizer 的 `offset_mapping` 把字符 span 映射到 token；如果 tokenizer 不支持 offset，就 fallback 到逐 token decode，此时只能识别 SID boundary 和 SID code，SID code 默认归到 history SID，interest SID 区分会变弱。

手工权重默认会对每条样本的有效 token 做 mean normalize：

```text
w_norm = w_raw / mean_valid(w_raw)
```

这样做的目的不是改变每条样本对 Hessian 的总贡献，而是改变同一样本内部不同 token 类型的相对权重。手工方案对应 mode：

- `weighted_gptq_fp8_w8a8`
- `weighted_gptq_fp8_w8a8_tail1`

### 4. token 加权方案二：梯度 token 粒度加权

梯度 token 粒度加权是 v2 的原始思路，实现在 `fake_quant_learnable/gradient_weights.py::collect_gradient_token_weight_batches_by_layer`。它不直接用手工 prompt role 权重，而是用 teacher-forced SID 预测 loss 的梯度来估计每个 token 对推荐 SID 决策的敏感度。

对每条 calib 样本，先从 ground truth 中取第一个 SID item，再取它 tokenization 后的第一个 token，也就是第一个 ground-truth `s_a` token id。然后在完整 prompt 的最后一个位置，也就是 `<|sid_begin|>` 位置上，计算 cross entropy：

```text
L = CE(logits[last_prompt_position], first_ground_truth_s_a)
```

在指定 layer 的输入 hidden state 上取梯度。第 l 层第 t 个 token 的敏感度定义为：

```text
s_{l,t} = mean_c | h_{l,t,c} * dL / dh_{l,t,c} |
```

代码中会先把模型参数 `requires_grad` 关掉，只保留被捕获的 hidden state 需要梯度。最小被选中层的输入会 detach 后重新 `requires_grad_(True)`，其他选中层用 `retain_grad()` 记录梯度。每条样本 forward 后对上述 CE loss backward，然后得到每层每个 token 的 sensitivity。

梯度权重还会做归一化和截断，默认 `GradientTokenWeightConfig` 为：


| 参数            | 默认值 | 作用                                                                    |
| ----------------- | -------: | ------------------------------------------------------------------------- |
| clip_percentile |   99.0 | 对有效 token 的 sensitivity 做 99 分位截断，避免极端 token 主导 Hessian |
| weight_floor    |   0.05 | 给有效 token 设置最小权重，避免某些 token 完全消失                      |
| normalize_mean  |   True | 有效 token 均值归一化到 1                                               |
| eps             |  1e-12 | 数值稳定                                                                |

v2 token 粒度版本的优点是最细，可以为同一 prompt 内每个 token 给不同权重；缺点也明显：梯度信号噪声大，calib 样本之间分布波动大，而且过细粒度的 token 权重可能让 Hessian 过拟合某些局部 token，而不是稳定地反映推荐 prompt 的结构。之前实验中它的整体效果没有超过手工 group 权重，基本接近 GPTQ+dynamic activation 的水平，但在 @32 上有一定提升。

### 5. token 加权方案三：v3 逐层梯度 group 加权

v3 是当前主线，也就是 `grad_weighted_gptq_fp8_w8a8` 和 `grad_weighted_gptq_fp8_w8a8_tail1` 现在实际使用的版本。代码路径是：

```text
collect_gradient_group_token_weight_batches_by_layer
  -> collect_gradient_token_weight_batches_by_layer
  -> group_token_weight_batches_by_layer
```

它分两步：

1. 先完全沿用 v2 的 teacher-forced gradient sensitivity，得到每层、每条 calib 样本、每个 token 的细粒度梯度权重。
2. 再使用和手工方案完全一致的 prompt role 分组，把 token 权重按 layer 和 group 聚合成 group mean，然后再回填到每个 token 上。

也就是说，v3 最终传给 GPTQ Hessian 的仍然是 per-token weight tensor，所以不需要改 GPTQ 接口；但同一个 layer 内，属于同一个 prompt role group 的 token 会共享一个由 calib 梯度统计出来的权重。形式上可以理解为：

```text
w_{l,t} = mean_{sample/token in group(t)} normalize(s_{l,t})
```

group 仍然是四类：

- `text`
- `history_sid`
- `interest_sid`
- `sid_boundary`

v3 和手工 group 的区别是：手工 group 权重是全层共享的一组固定 prior；v3 的 group 权重是用 calib 梯度算出来的，而且每一层都可以不同。v3 和 v2 token 粒度的区别是：v2 保留每个 token 的细粒度波动，v3 把这些波动压缩成 layer-wise group 均值。因此 v3 是介于“全手工 group prior”和“完全 token 粒度梯度”之间的折中版本：既保留推荐 prompt role 的结构约束，又让不同 layer 的重要性由梯度数据决定。

当前实现是纯梯度 group 权重，没有和手工权重做 shrinkage 或 prior 融合。之前 probe 显示，v3 相比手工权重通常会抬高 `history_sid`，降低部分层的 `text`，并在中后层显著提高 `interest_sid`，这说明它确实捕捉到了和手工 prior 不同的 layer-wise 推荐信号。

### 6. activation per-token dynamic quantization

activation dynamic quantization 在 `fake_quant_learnable/quant.py::activation_per_token_qdq_forward`。对 activation tensor 的最后一维 hidden dimension 做 per-token absmax：

```text
scale_t = max_c |x_{t,c}| / 448
q_t = clamp(x_t / scale_t, -448, 448).to(torch.float8_e4m3fn)
x_qdq_t = q_t.float() * scale_t
```

输出再 cast 回原始 dtype，所以 fake quant 下模型后续计算仍在原 dtype 中进行。这个逻辑模拟的是真实部署中 activation runtime per-token scale + FP8 E4M3 的数值误差。

当前默认 `act_quant_mode="shared_input"`，因此不是每个 Linear 各自独立 quantize 输入，而是对 Qwen3 block 中共享同一个输入的 Linear 做一次 shared activation QDQ：

- attention 中 `q_proj/k_proj/v_proj` 共享 `hidden_states` 的一次 QDQ。
- attention 中 `o_proj` 对 attention output 做一次 QDQ。
- MLP 中 `gate_proj/up_proj` 共享 MLP 输入的一次 QDQ。
- MLP 中 `down_proj` 对 `act(gate) * up` 的中间结果做一次 QDQ。

这样更接近真实 fused/shared-input 部署，也避免 q/k/v 或 gate/up 对同一份输入重复产生不同 fake quant 噪声。

### 7. tail1 BF16 activation 保护

tail1 保护在 `fake_quant_learnable/quant.py::activation_per_token_qdq_forward_tail_protected` 和 `GPTQFakeQuantLinear.activation_tail_tokens` 中实现。逻辑是：先正常做 per-token activation QDQ，然后把最后 `tail_tokens` 个序列位置恢复为原始 activation：

```text
x_qdq[..., -tail_tokens:, :] = x_original[..., -tail_tokens:, :]
```

当前 tail1 mode 中 `activation_tail_tokens=1`。它的含义是：

- prefill 阶段：保护 prompt 的最后一个 token。由于 item prediction prompt 后面会 append `<|sid_begin|>`，所以这里保护的是最后的 `<|sid_begin|>` activation。
- decode 阶段：KV-cache decode 输入通常是 `[batch*beam, 1, hidden]`，序列长度为 1；tail1 会覆盖整个 decode step，因此 decode 阶段 activation 等价于保持 BF16/original dtype，不做 activation FP8 QDQ。

shared-input 路径里，`_shared_prepare_input` 会从第一个 quant Linear 读取 `activation_tail_tokens`。如果大于 0，就对 shared activation 输入执行 tail-protected QDQ。因此 q/k/v、gate/up、o/down 这些 shared-input 位置也会遵循同样的 tail1 保护，而不是只在普通 per-linear wrapper 中生效。

需要强调的是：tail1 只保护 activation，不取消 GPTQ weight 量化。也就是说，当前最优组合仍然是 GPTQ FP8 fake-quant weight；只是 prefill 最后 token 和 decode token 的 activation 不承受 FP8 dynamic activation QDQ 误差。从推荐生成机制看，这两个位置很关键：`<|sid_begin|>` 直接决定第一个 SID token `s_a` 的分布，decode 阶段则直接决定后续 `s_b/s_c` 和 beam 扩展。

### 8. 当前方案的定位

当前主线方法可以概括为：

```text
weight side:
  GPTQ FP8 E4M3 weight fake quant
  + Hessian token weighting
  + v3 layer-wise gradient group prompt-role weights

activation side:
  dynamic per-token FP8 E4M3 fake quant
  + Qwen3 attention/MLP shared-input quantization
  + tail1 BF16/original activation protection
```

它和 naive W8A8 的主要区别不在 activation quant 公式，而在 weight 量化校准目标：naive weight 是直接按 output channel scale 吸附到 FP8 网格；GPTQ 会用校准激活 Hessian 做误差补偿；token 加权 GPTQ 又进一步把 Hessian 从普通 `X^T X` 改成推荐 prompt token-aware 的 `X^T diag(w) X`。因此它本质上是在 weight PTQ 阶段把推荐任务里更关键的 token 方向显式写进优化目标。

当前最应该继续报告和复跑的主线实验是：

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad \
  --mode grad_weighted_gptq_fp8_w8a8_tail1 \
  --model_path /home/guowei/OneRec-1.7B/ \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --output_dir fake_quant_learnable/results/<experiment_name> \
  --device cuda:0 \
  --calib_sample_size 1024 \
  --eval_sample_size full \
  --overwrite \
  --evaluate
```

对于对照实验，需要同时保留：

- `baseline_w8a8`：naive W8A8。
- `gptq_fp8_w8a8`：不加 token weight 的 GPTQ。
- `weighted_gptq_fp8_w8a8`：手工 group 权重，不带 tail1。
- `weighted_gptq_fp8_w8a8_tail1`：手工 group 权重 + tail1。
- `grad_weighted_gptq_fp8_w8a8`：v3 梯度 group 权重，不带 tail1。
- `grad_weighted_gptq_fp8_w8a8_tail1`：v3 梯度 group 权重 + tail1。

这样可以把收益拆成三块看：GPTQ weight 量化本身的收益、token-aware Hessian 的收益、tail1 activation 保护的收益。

## 真实 FP8 推理落地路线 / 2026-06-17

当前 `fake_quant_learnable` 的主线实现仍然是 fake quant：weight 和 activation 都会模拟 FP8 E4M3 的量化误差，但 Linear 计算本身仍然走 `F.linear`，输入和权重在 matmul 时不是常驻 FP8 数据格式。因此它可以评估精度影响，但不能直接证明真实部署时延收益。

下一步真实 FP8 runtime 的目标是把离线量化结果转成真正可执行的低精度路径：

```text
offline:
  FP/BF16 weight -> FP8 weight + FP32 scale

runtime:
  BF16 activation -> dynamic FP8 activation + FP32 scale
  FP8 activation x FP8 weight -> BF16 output
```

在当前 PyTorch 环境中，`torch._scaled_mm` 可以表达我们现有 scale 粒度：

- activation per-token scale: `[batch, seq, 1]` flatten 后变成 GEMM 侧的 `scale_a = [M, 1]`。
- weight per-output-channel scale: Linear weight `[N, K]` 的 scale `[N, 1]` 转成 GEMM 侧的 `scale_b = [1, N]`。
- `_scaled_mm` 的输入为 `A_fp8=[M,K]` 和 `B_fp8=[K,N]`，输出使用 BF16。

因此，scale reshape 和 FP32 cast 本身不是主要难点。主要工程难点在于：FP8 weight 常驻保存、`_scaled_mm` 所需的权重 layout、activation dynamic quant 的额外开销、shared-input/fused projection、tail1 的 BF16 分流，以及 beamsearch/generate 端到端开销。

### Stage 1: real naive W8A8

第一阶段先实现最朴素的真实 FP8 W8A8 runtime，不引入 GPTQ、token weight 或 tail1。目标是验证 real FP8 backend 本身是否能跑通，以及 fake quant 的数值结果能否迁移到真实 `_scaled_mm` 路径。

实现要点：

- 对每个 Linear 的原始 weight 做 per-output-channel FP8 quant。
- 保存 `weight_fp8` 和 `weight_scale`，而不是只保存 `weight_qdq`。
- forward 中把 activation flatten 为 `[M, K]`，按 per-token 动态计算 `scale_a=[M,1]`，cast 到 FP8。
- 使用 `torch._scaled_mm(x_fp8, weight_fp8.t(), scale_a, scale_b, out_dtype=torch.bfloat16)` 计算。
- 第一版每个 Linear 独立实现，不强制做 q/k/v 或 gate/up fusion。
- 不做 tail1，避免一开始引入 FP8/BF16 混合路径。

成功标准：

- 能完整跑一次 OneRec eval。
- real naive W8A8 的指标和当前 fake `baseline_w8a8` 接近。
- 单 Linear benchmark 中 FP8 GEMM 相比 BF16 matmul 有明确收益。
- 端到端耗时至少不能明显劣化；如果劣化，需要 profile activation cast、layout 转换和 Python 调度。

### Stage 2: real GPTQ 原版

第二阶段保持 Stage 1 的 runtime 不变，只把 weight 来源从 naive RTN 替换为 GPTQ 量化后的 FP8 weight。这里的关键是离线 GPTQ 不再只返回 `weight_qdq`，而是额外导出：

```text
weight_fp8
weight_scale
optional weight_qdq_for_debug
```

runtime 仍然统一走 `RealFP8Linear`。这样可以把算法收益和 runtime 实现解耦。

成功标准：

- real GPTQ 的指标接近当前 fake `gptq_fp8_w8a8`。
- real GPTQ 相比 real naive W8A8 有可观察的精度优势。
- 不因为 GPTQ weight 来源改变而引入额外 forward 开销。

### Stage 3: real weighted GPTQ

第三阶段再接入 token-aware GPTQ，包括手工 group 权重和 v3 逐层梯度 group 权重。这个阶段理论上不需要改变 runtime，因为 token weighting 只影响离线 GPTQ 的 Hessian 和权重搜索，最终导出的仍然是同一种格式：

```text
weight_fp8 + weight_scale
```

推荐保留的 ablation 顺序：

- `real_naive_w8a8`
- `real_gptq_fp8_w8a8`
- `real_weighted_gptq_fp8_w8a8`
- `real_grad_weighted_gptq_fp8_w8a8`

成功标准：

- weighted GPTQ 的 real FP8 指标趋势和当前 fake quant 结果一致。
- 手工权重、v3 梯度 group 权重的收益能在真实 runtime 下复现。
- runtime 文件和量化算法文件边界清晰：加权逻辑不侵入 `RealFP8Linear`。

### Stage 4: real weighted GPTQ + tail1

最后再加入 tail1，因为 tail1 会引入混合精度路径：

```text
non-tail token:
  FP8 activation + FP8 weight -> _scaled_mm

tail token:
  BF16/original activation + quantized/dequantized or BF16 fallback weight -> BF16 matmul
```

在当前推荐生成场景中，tail1 的语义尤其特殊：

- prefill 阶段保护最后一个 `<|sid_begin|>` token。
- decode 阶段常见输入长度为 1，因此严格 tail1 会让 decode token 的 activation 不走 FP8 dynamic quant。

因此 tail1 很可能提高精度，但会削弱 decode 阶段的 FP8 速度收益。它应该放在最后实现，避免在还没验证 real FP8 backend 前就把性能问题复杂化。

不过经过 real quant phase timing 后，decode 阶段应单独看待：**decode activation 保留 A16 是当前实现下比较确定的工程收益点**。原因有两个：

1. 推荐 SID 生成的 decode 很短。OneRec 当前 `max_new_tokens=3`，第一个 SID token 来自 prefill logits，真正 `seq_len=1` 的 autoregressive decode 只有后续约 2 次；因此 decode phase 本身只占端到端 generate 的小比例。
2. decode 阶段的矩阵乘法规模太小。beam 展开后典型 Linear 输入为 `[batch * beam, 1, hidden]`，flatten 后 `M≈32`。这种小 `M` 场景下，FP8 `_scaled_mm` 很难吃到大矩阵乘法红利，反而要额外承担 dynamic activation quant、scale 处理和 kernel launch 开销。

因此，decode 走 W8A16 并不是因为 W8A16 理论算力强于 W8A8，而是因为在 OneRec 短 decode、小矩阵的实际路径中，绕过 FP8 activation quant 和小规模 `_scaled_mm` 更划算。当前小样本 phase timing 中，decode W8A16 让 decode forward 本身约快 15%，端到端 generate 约快 1%。这个收益不大，但方向稳定；主要加速来源仍然应来自 prefill 大矩阵 FP8 GEMM 和 fused dynamic activation quant。

成功标准：

- real tail1 指标接近当前 fake `*_tail1` 的提升趋势。
- prefill 非 tail token 仍然走 FP8 `_scaled_mm`。
- tail token 的 BF16 fallback 路径数值明确、开销可测。
- 报告中单独拆分 prefill、decode、beamsearch/topk、lm_head 等耗时，避免只用总耗时判断 FP8 是否有效。

### 推荐执行顺序

当前最稳的推进方式是：

```text
1. real_naive_w8a8
2. real_gptq_fp8_w8a8
3. real_weighted_gptq_fp8_w8a8 / real_grad_weighted_gptq_fp8_w8a8
4. real_grad_weighted_gptq_fp8_w8a8_tail1
```

这个顺序的好处是每一步只新增一个变量：先验证 runtime，再验证 GPTQ，再验证 token-aware Hessian，最后验证 tail1。这样可以避免把真实 FP8 kernel、GPTQ 量化收益、token weighting 收益和 tail1 精度/速度 tradeoff 混在一起。

## 当前方案的叙事逻辑

> 目前方案：weight使用GPTQ的token加权方案，token加权的粒度是layer token group，加权权重为校准集teacher-forced SID 预测 CE loss 的梯度聚合；activation使用朴素的dynamic activation quantization，此外，在decode阶段为了保证精度，使用w8a16，在prefill阶段，对activation最后一个token做精度保留。在量化位置上，对nn.linear进行替换，包括attention q/k/v/o proj. + ffn up/down/gate proj.

我的初步叙述：
经典大模型量化算法GPTQ在计算中把所有token一视同仁，但是在推荐大模型场景下存在明显的token差异，其中推荐token相对其他token的影响度较低，直接使用GPTQ效果不好，精度下降严重。故考虑改进成token-aware加权版本的GPTQ方法。
又因为GPTQ为weight-only量化，理论上主要节省权重存储/访存，为保证推理时的计算时延（想要用上fp8×fp8的GEMM），故在推理时对激活值使用per-token量化
同时为了保证推荐场景下对精度的严格要求，保留了prefill阶段last token和decode阶段的推理激活值为bf16。这里需要区分两类开销：prefill last token 的 A16 分支主要服务于精度保护，可能带来额外开销；decode A16 则因为推荐场景 decode 次数短、矩阵规模小、FP8 `_scaled_mm` 吃不到大矩阵红利，在当前 real runtime 中反而是小幅时延收益点。

### 中文 Abstract 版本 / 2026-06-24（摘要稿）

现有大模型量化方法主要面向通用文本生成场景，通常默认校准 token 对量化误差的贡献近似均匀。**但推荐大模型的目标并不是自由文本生成，而是在用户历史、兴趣描述和候选语义条件下生成结构化 SID token 序列**。推荐 prompt 中的自然语言、历史 SID、兴趣 SID、SID 边界承担不同功能；同时，SID 以 `s_a/s_b/s_c` 三段式结构生成，首个 SID token 会约束后续 SID code 的组合空间，后续 decode token 也必须沿固定格式完成同一 item 的语义 ID。因此，推荐大模型量化的核心问题不是简单压低整体量化误差，而是如何把有限的低精度误差预算分配到对 SID 生成最关键的 token 角色和生成位置上。

基于这一观察，我们提出一种面向推荐生成的 FP8 W8A8 量化方案。weight 侧以 GPTQ 为基础，将等权校准目标扩展为 token-aware 形式，并利用校准集 teacher-forced SID 预测梯度估计不同 prompt role 在不同层的重要性，从而使权重量化更关注对推荐生成敏感的输入方向。activation 侧采用 per-token dynamic FP8 quantization，以对齐真实 W8A8 FP8 GEMM 的部署路径；同时，对 `<|sid_begin|>` 和 decode 阶段等关键生成位置保留 BF16 activation，以抑制结构化 SID 生成中的误差放大。这样，weight 量化和 activation 量化分别从离线校准目标和在线推理路径两侧，共同服务于推荐生成中的敏感性保护。

当前实验表明，该方法在 FP8 W8A8 数值约束下能够显著恢复 naive W8A8 的精度损失，并在 product/video 等推荐任务上达到接近全精度模型的输出精度。后续真实 FP8 runtime 部署的目标是在保持该精度趋势的同时获得约 1.3x 的端到端推理时延加速。区别于主要沿用通用 LLM PTQ 或系统级 FP8 推理优化的已有路线，我们将该方法定位为推荐大模型领域中首个将推荐 SID 生成任务结构显式引入量化校准目标的模型量化算法。

## MLLM/VLM 量化扩展调研 list / 2026-06-24

本节整理 MLLM/VLM 量化论文和开源实现，重点看它们如何把多模态模型中的 token/modality heterogeneity 转成 PTQ 算法设计。总体结论是：这些工作大多没有发明全新的低层量化算子，而是在校准目标、Hessian、smoothing scale、outlier channel、token 保护或静态 scale 上引入模态/ token 结构。对 OneRec 来说，最自然的迁移不是把视觉 token 直接类比成 SID token，而是把 `text / history_sid / interest_sid / sid_boundary / generation slot` 当成推荐模型内部的异质 token group。

### 1. 已调研论文和实现状态


| 方法                                                                 |      年份 | 核心思路                                                                                                                                                                                | 开源实现状态                                                                                                                                             | 对 OneRec 的参考价值                                                                                                                                                           |
| ---------------------------------------------------------------------- | ----------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MBQ: Modality-Balanced Quantization for Large Vision-Language Models |      2024 | 发现 vision/text token 的量化敏感性不同，在 calibration reconstruction loss 中按模态重加权，避免大量但低敏感的视觉 token 主导量化参数。支持 weight-only 和 W+A。                        | 已开源：https://github.com/thu-nics/MBQ 。代码基于`qmllm`，README 显示支持 `mbq/awq/smoothquant/rtn`，评估时使用 `--pseudo_quant`，并有 W3 kernel 结果。 | 很高。我们当前的 token-aware GPTQ 与 MBQ 的思想最接近：普通 GPTQ 等权 token 不适合异质 token 场景，应把 Hessian/reconstruction loss 改成 SID role aware。                      |
| VLMQ: Efficient PTQ for VLMs via Hessian Augmentation                |      2025 | 认为视觉 token 多且冗余，普通 GPTQ/AWQ 的 Hessian 被等权 token 统计污染；用 token-level importance 增强 Hessian，且通过轻量 block-wise backward 估计重要性。                            | 未查到稳定官方仓库。论文页面：https://arxiv.org/abs/2508.03351 。                                                                                        | 很高。它几乎直接支持我们的`X^T diag(w) X` 路线。OneRec 可以把 VLMQ 的视觉 token redundancy 叙事替换成推荐 prompt 中不同 SID role 对最终 SID 生成贡献不均。                     |
| QIG: Quantization-Aware Integrated Gradients                         |      2026 | 用 Integrated Gradients 估计 token sensitivity，把 MBQ/VLMQ 的粗模态粒度推进到 token 级，反映跨模态和模态内 token 的动态差异。                                                          | 已开源：https://github.com/ucas-xiang/QIG 。仓库包含`qmllm`、`main_quant.py`、`inference.py`，README 标注 CVPR 2026 官方实现。                           | 中高。适合参考“细粒度重要性估计”的实验设计，但不建议主线直接上 IG；OneRec calibration set 小，IG 成本高且噪声可能大。更稳的是保留当前 layer-wise group gradient weight。     |
| MASQuant: Modality-Aware Smoothing Quantization                      |      2026 | 从 SmoothQuant 出发，指出共享 smoothing scale 在 MLLM 中有 smoothing misalignment；为不同模态学习/估计不同 smoothing factor，并用 cross-modal compensation 处理多模态计算不变性问题。   | 已开源在 Alibaba EfficientAI：https://github.com/alibaba/EfficientAI ，README 中有`masquant` 目录和 MASQuant 入口。                                      | 中等。可启发 SID-aware SmoothQuant/LET，但我们已有 SQ 实验收益有限，而且多套 activation scale 会增加 real quant 部署复杂度。适合作为附录/负结果支撑，不建议当前主线。          |
| MQuant: Full Static Quantization for MLLMs                           |      2025 | 针对 dynamic activation quant 开销大、视觉/文本 token 分布不同的问题，提出 modality-specific static scale，并通过 token reorder/attention-invariant switching 让静态 scale 更适合推理。 | 论文称代码将发布，未查到可用仓库。论文页面：https://arxiv.org/abs/2502.00425 。                                                                          | 中高，主要影响 real quant。它说明如果想要端到端时延收益，dynamic activation quant 可能需要被 static/typed scale 替代。但这会改变我们当前 naive W8A8 路径，暂时不应抢算法主线。 |
| SplitQ: Breaking Modality Heterogeneity in Low-Bit Quantization      |      2026 | 发现跨模态异质性不是均匀分布在所有 channel，而是少量 modality-specific outlier channel；提出 modality-specific outlier channel decoupling 和轻量 adaptive calibration branch。          | 仓库公开：https://github.com/EMVision-NK/SplitQ 。当前仓库较轻，主要是论文代码入口，完整度需要继续观察。                                                 | 中高。对 tail1 的替代方案有启发：不要整段 token BF16 保护，而是定位 SID-sensitive channel，做少量 channel protection / compensation，可能更时延友好。                          |
| MorphoQuant: Modality-Aware Quantization for Omni-modal LLMs         |      2026 | 面向 omni-modal，提出 distribution-aware bias compensation，把长尾 outlier 吸收到 channel-wise bias，同时优化量化函数/量化网格以保留不同模态分布形态。                                  | 未查到可用代码。论文页面：https://arxiv.org/abs/2606.04349 。                                                                                            | 中等。可借鉴 bias compensation 思路，用于替代 tail token BF16，但实现和验证成本较高。                                                                                          |
| VEQ: Modality-Adaptive Quantization for MoE VLMs                     |      2026 | 面向 MoE VLM，同时考虑 modality token heterogeneity 和 expert heterogeneity；用 expert activation frequency 与 token-expert affinity 构造增强 Hessian。                                 | 仓库公开但代码未发布：https://github.com/guangshuoqin/VEQ ，README TODO 仍写着 release code。                                                            | 中等。OneRec 目前不是 MoE，直接相关性不强；但“按模块/路径使用频率给 Hessian 加权”的思想可以转成 layer/module sensitivity weighting。                                         |
| QAPruner: Quantization-Aware Vision Token Pruning                    |      2026 | 研究 token pruning 与 PTQ 的耦合，指出普通语义 pruning 可能删掉对量化稳定性重要的 outlier token；用 quantization error + outlier intensity + semantic score 共同选 token。              | 未查到官方代码。论文页面：https://arxiv.org/abs/2604.02816 。                                                                                            | 中低。它启发 beam/candidate pruning 和量化耦合，但会把问题从 PTQ 扩到 pruning/search，当前不建议主线推进。                                                                     |
| Towards Understanding Best Practices for Quantization of VLMs        |      2026 | 系统比较 GPTQ/AWQ 作用于 vision encoder、LLM、connector 等组件时的效果，强调组件敏感性和任务差异。                                                                                      | 已开源：https://github.com/gautomdas/mmq 。仓库含 AWQ/GPTQ、BLIP/LLaVA 任务和评估脚本。                                                                  | 高。我们也需要做 OneRec 的 component sensitivity：embedding/lm_head/attention/MLP/早中晚层/SID 输出相关位置分别量化，给主方法提供定位依据。                                    |
| Quant Experts: Token-aware Adaptive Error Reconstruction with MoE    |      2026 | 认为重要 channel 的分布会随 modality 和 token 变化；把重要 channel 分为 token-independent 与 token-dependent，并用 shared/routed low-rank expert 补偿量化误差。                         | 未查到官方代码。论文页面：https://arxiv.org/abs/2602.24059 。                                                                                            | 中等。说明 token-dependent compensation 是一个方向，但引入 routed adapter 会接近小规模训练/额外推理分支，不适合当前 PTQ 简洁主线。                                             |
| SliM-LLM / TaCQ 等任务敏感混合精度工作                               | 2024-2025 | 虽非 MLLM 主线，但都强调用 salience、gradient 或 task circuit 选择高精度保留对象，而不是所有权重同等量化。                                                                              | SliM-LLM 页面：https://arxiv.org/abs/2405.14917 ；TaCQ 页面：https://arxiv.org/abs/2504.07389 。                                                         | 中等。可作为“任务敏感量化”相关工作，但不要喧宾夺主。OneRec 更适合保持无额外在线分支的 weighted GPTQ。                                                                        |

### 2. MLLM 量化论文的共性

这些论文可以粗略分成五类：

1. **modality/token 加权校准**：MBQ、VLMQ、QIG、VEQ。核心是普通 PTQ 的等权 calibration 不适合异质 token，应该让 Hessian 或 reconstruction loss 看见 token 重要性。
2. **modality-aware smoothing / scale**：MASQuant、MQuant。核心是不同模态 activation 分布不同，单一 scale 或 per-token dynamic scale 都可能不是最优；前者影响精度，后者影响时延。
3. **outlier channel 分离或补偿**：SplitQ、MorphoQuant、Quant Experts。核心是异质性并不只在 token 维度，也会投影到少数 channel；可以保护/补偿关键 channel，而不是粗暴保护整段 token。
4. **pruning 与 quantization 协同**：QAPruner。核心是被剪掉的 token 可能对量化稳定性重要，token selection 需要同时考虑语义和量化误差。
5. **系统化 best practice / component sensitivity**：Best Practices for VLM Quantization。核心是先找清楚哪个组件对任务指标最敏感，再决定在哪里量化、哪里保留高精度。

对 OneRec 来说，第一类最适合当前 fake quant 主线，第三类最适合作为 tail1 的后续替代，第二类更偏 real quant 时延优化，第四类暂时不建议做。

### 3. 对当前 OneRec 方案的判断

当前 `grad_weighted_gptq_fp8_w8a8_tail1` 和 MLLM 文献的关系可以表述为：

```text
MLLM:
  vision/text token heterogeneity
  -> modality/token-aware calibration
  -> weighted reconstruction or enhanced Hessian

OneRec:
  text/history_sid/interest_sid/boundary/generation-slot heterogeneity
  -> SID-role-aware calibration
  -> weighted GPTQ Hessian for FP8 weight quantization
```

也就是说，我们现在最像 MBQ + VLMQ 的推荐模型版本，而不是 OmniQuant/MASQuant 的 smoothing 版本。这个定位比“直接优化 top-k stability”更自然，也比“重新学习 LET/LWC”更贴合目前实验结果。

需要注意一个表述风险：文档前面写到“推荐 token 相对其他 token 的影响度较低，直接使用 GPTQ 效果不好”。这句话应谨慎使用。更稳的说法是：

> 推荐 prompt 中不同 token role 对最终 SID 生成的敏感性和数量分布不均，等权 Hessian 可能被 token 数量或非关键方向主导；因此需要用 SID teacher-forced 信号估计 layer-wise role importance，对 GPTQ Hessian 做校准期重加权。

这样不会陷入“SID token 梯度低所以要高权重还是低权重”的矛盾。我们的算法本质不是简单抬高某类 token，而是让 calibration Hessian 反映推荐任务相关的输入方向。

### 4. 下一步建议

短期主线建议保持：

```text
SID-role-aware / gradient group weighted GPTQ
  + dynamic activation fake quant
  + tail1 作为精度保护 ablation
  + real quant 只做 latency sanity check
```

优先补三组实验，而不是马上引入更复杂方法：

1. **component sensitivity**：参考 VLM best practice，分别测试 attention、MLP、lm_head、早/中/晚层、skip last、tail1 的精度影响。目标是证明 OneRec 的量化敏感性集中在哪里。
2. **Hessian weighting ablation**：普通 GPTQ、手工 group、纯梯度 token、layer-wise gradient group、不同 calib size。目标是证明收益来自推荐结构感知的 Hessian，而不是偶然超参。
3. **tail1 替代探索**：参考 SplitQ/MorphoQuant，尝试少量 SID-sensitive channel protection 或 bias compensation，目标是在保留 tail1 精度收益的同时减少 real quant 时延损失。

不建议当前推进的方向：

- 不建议把 MASQuant/MQuant 式 modality-specific activation scale 作为下一步主线。它和 real quant 强绑定，工程复杂度高，而且前面 SmoothQuant 已经显示收益有限。
- 不建议直接复现 QIG 的 Integrated Gradients。它的解释性强，但在 OneRec 小 calib set 上可能噪声更大，计算成本也高。
- 不建议主攻 QAPruner/beam pruning。它会改变推荐生成/搜索流程，容易偏离 PTQ 算法主线。

因此，当前最有投稿潜力的叙事仍然是：

> MLLM PTQ 已经证明异质 token 不能等权校准；OneRec 推荐大模型虽然不是传统多模态模型，但 SID token 和固定 SID 生成结构引入了推荐任务特有的 token role heterogeneity。我们将这种结构显式写入 GPTQ Hessian，得到一种无额外在线开销的推荐感知 FP8 PTQ 方法。

## 新方案：SAC-GPTQ / SID-aware Activation-Compensated GPTQ / 2026-06-24

### 1. 当前困境

当前主线 `grad_weighted_gptq_fp8_w8a8_tail1` 的效果虽然可观，但算法结构仍然比较割裂：

- weight 侧是 GPTQ 的 token 加权版本，本质仍是 weight-only PTQ；
- activation 侧是标准 per-token dynamic FP8 QDQ，本身没有新算法；
- tail1 用 BF16 activation 保护关键生成位置，效果上像一个后处理补丁；
- weight calibration 并没有显式看到推理时 activation quantization 产生的输入扰动。

因此，当前方法可以解释为“推荐 token-aware GPTQ + 常规 activation quant + 关键 token 保护”，但还不是一个真正统一的 W+A PTQ 目标。

SAC-GPTQ 的目标是解决这个割裂：**在离线搜索 FP8 weight 时，直接把推理时的 activation QDQ 放进 weight 量化目标，让 weight 量化主动补偿 activation 量化误差**。如果成功，它有机会替代 tail1，保持无额外在线开销。

### 2. 相关工作边界

SAC-GPTQ 不是凭空引入“联合 W/A”这个大方向，已有工作可以支撑它的合理性，但它和它们的落点不同：

- GPTQ 的二阶目标是 weight-only：用校准输入 Hessian 近似 `X(W - W_q)^T` 的输出误差，未显式建模 activation quantization。论文：https://arxiv.org/abs/2210.17323
- MBQ / VLMQ 说明异质 token 不应等权参与 PTQ calibration，分别从 modality-balanced reconstruction 和 token-level Hessian augmentation 改造校准目标。MBQ：https://arxiv.org/abs/2412.19509 ，VLMQ：https://arxiv.org/abs/2508.03351
- MQuant 说明 W+A 推理中 dynamic activation quantization 的代价和模态/token scale 设计很关键，但它主要改 activation scale 和静态量化路径。论文：https://arxiv.org/abs/2502.00425
- CoQuant 说明输出误差由 weight 和 activation quantization noise 共同驱动，joint weight-activation modeling 是合理方向。论文：https://arxiv.org/abs/2604.26378

SAC-GPTQ 的区别是：它不做额外高精度子空间、不引入在线 adapter、不改变推理 kernel，而是把 `Q_a(X)` 直接写进 GPTQ 的离线 weight target，并用 OneRec 的 SID role / SID slot 权重约束补偿方向。

### 3. 数学目标

考虑一个 Linear 层，原始 BF16 forward 为：

$$
Y = X W^T

$$

其中：

- $X \in \mathbb{R}^{N \times d}$ 是校准样本中该 Linear 的输入；
- $W \in \mathbb{R}^{m \times d}$ 是原始权重；
- $Y \in \mathbb{R}^{N \times m}$ 是 teacher 输出；
- $N$ 是 batch、sequence、slot 展平后的 token 行数。

真实 W8A8 推理并不是使用 $X$，而是使用 activation QDQ 后的输入：

$$
X_q = Q_a(X)

$$

如果继续用普通 GPTQ 优化：

$$
\min_{\hat W}
\left\|XW^T - X\hat W^T\right\|_F^2

$$

那么 weight 搜索看到的输入和真实推理输入不一致。SAC-GPTQ 改成直接优化：

$$
\min_{\hat W}
\left\|XW^T - X_q\hat W^T\right\|_{\Omega}^{2}

$$

其中：

$$
\Omega = \mathrm{diag}(\omega_1, \omega_2, \ldots, \omega_N)

$$

$\tau_i$ 表示第 $i$ 行 activation 对应的 token role 或 generation slot，$\beta(\tau_i)$ 表示 SID-aware 权重，则：

$$
\omega_i = \beta(\tau_i)

$$

更稳的版本可以加入原始权重正则，避免补偿目标过度偏离 BF16 weight：

$$
\min_B
\left\|XW^T - X_qB\right\|_{\Omega}^{2}
+
\lambda \left\|B - W^T\right\|_F^2

$$

这里 $B \in \mathbb{R}^{d \times m}$ 是连续补偿权重的转置形式。该目标是标准加权 ridge least squares，对 $B$ 求导并令梯度为 0：

$$
X_q^T\Omega X_q B - X_q^T\Omega XW^T + \lambda(B - W^T) = 0

$$

得到闭式解：

$$
B_c =
\left(X_q^T\Omega X_q + \lambda I\right)^{-1}
\left(X_q^T\Omega X + \lambda I\right)W^T

$$

令：

$$
H_{qq} = X_q^T\Omega X_q

$$

$$
H_{qx} = X_q^T\Omega X

$$

则：

$$
B_c = (H_{qq} + \lambda I)^{-1}(H_{qx} + \lambda I)W^T

$$

维度检查：

- $H_{qq} \in \mathbb{R}^{d \times d}$；
- $H_{qx} \in \mathbb{R}^{d \times d}$；
- $W^T \in \mathbb{R}^{d \times m}$；
- $B_c \in \mathbb{R}^{d \times m}$；
- 最终待量化 target weight 为 $W_c = B_c^T \in \mathbb{R}^{m \times d}$。

### 4. 与 GPTQ 的连接

上面的连续问题不是最终量化结果，因为推理时 weight 仍然必须落在 FP8 网格上。关键是：加权 ridge 目标可以改写成围绕 $B_c$ 的二次型：

$$
\left\|XW^T - X_qB\right\|_{\Omega}^{2}
+
\lambda \left\|B - W^T\right\|_F^2
=
\left\|B - B_c\right\|_{H_{qq}+\lambda I}^{2}
+
C

$$

其中 $C$ 与 $B$ 无关。因此，离散 FP8 搜索可以近似为：

$$
\min_{\hat B \in \mathcal{Q}}
\left\|\hat B - B_c\right\|_{H_{qq}+\lambda I}^{2}

$$

这和 GPTQ 的形式一致，只是：

```text
普通 GPTQ:
  target weight = W
  Hessian = X^T Omega X

SAC-GPTQ:
  target weight = W_c
  Hessian = X_q^T Omega X_q + lambda I
```

所以第一版实现不需要重写 GPTQ 主循环，只需要把 GPTQ 的输入从 `(W, H)` 扩展为：

```text
(target_weight = W_c,
 hessian = Hqq + lambda I,
 scale_source = original W or W_c)
```

这里建议第一版使用 `scale_source = original W`，原因是当前 fake/real GPTQ 的 FP8 scale 语义都更接近“原始权重 per-output-channel absmax scale”。如果直接用 $W_c$ 计算 scale，补偿目标可能放大少数 channel 的 range，导致 scale 变粗，反而损害其他列。后续可以把 `original-scale` 和 `target-scale` 作为 ablation。

### 5. 合理性检查

SAC-GPTQ 至少满足以下数学边界：

1. 如果没有 activation quantization，即 $X_q = X$，则：

$$
B_c = (X^T\Omega X + \lambda I)^{-1}(X^T\Omega X + \lambda I)W^T = W^T

$$

此时退化为普通 weighted GPTQ。

2. 如果 $\lambda$ 足够大，则 $B_c$ 会被拉回 $W^T$，补偿强度趋近于 0，避免小校准集下过拟合 activation noise。
3. 如果 $\lambda = 0$ 且 $X_q$ 满秩，$B_c$ 是在量化 activation 输入下最小化 teacher Linear 输出误差的最小二乘解。
4. 推理阶段没有额外参数、分支或 BF16 fallback；最终仍然导出 FP8 weight，activation 仍然走当前 dynamic FP8 QDQ。所有额外计算只发生在 calibration 阶段。
5. 它把当前割裂的三件事合并到一个目标中：

```text
weight quantization error
activation quantization error
SID role / generation slot sensitivity
```

因此，它比“weighted GPTQ + 常规 activation quant + tail1”更像一个统一算法。

### 6. OneRec 中的 SID-aware Omega

SAC-GPTQ 的 $\Omega$ 不应只沿用简单 token group 权重。因为它的目标是补偿 activation quantization，所以最关键的不是“哪类 token 语义重要”，而是“哪类 token/slot 的 activation QDQ 误差更应该被 weight 离线吸收”。

建议第一版使用两级权重：

```text
role weight:
  text
  history_sid
  interest_sid
  sid_boundary

slot weight:
  prefill_last_sid_begin
  decode_s_a
  decode_s_b
  decode_s_c
```

校准输入也应从只看 prefill prompt 扩展为 teacher-forced SID slot：

```text
slot a:
  prompt + <|sid_begin|>
  target = s_a

slot b:
  prompt + <|sid_begin|> + s_a
  target = s_b

slot c:
  prompt + <|sid_begin|> + s_a + s_b
  target = s_c
```

这样 SAC-GPTQ 才能学习到 decode 阶段 activation QDQ 对后续 SID token 的影响。为了避免重复 prompt token 被三次 teacher-forced prefix 过度计数，建议每个 slot 内做样本级归一化，或者只给当前 slot 的最后一个 token 较高权重，prompt/history token 保持较低背景权重。

### 7. 实现可行性

当前代码已经具备 SAC-GPTQ 的大部分基础：

- `fake_quant_learnable/quant.py` 已有 `activation_per_token_qdq_forward`，可得到 $X_q$；
- `fake_quant_learnable/gptq.py::collect_gptq_hessians` 已支持 token weights，可扩展为收集 `Hqq` 和 `Hqx`；
- `gptq_fp8_quantize_weight` 已支持给定 Hessian 做 FP8 GPTQ，只需要允许 target weight 和 scale source 分离；
- real quant 侧不需要新增 runtime 分支，因为 SAC-GPTQ 只改变离线 weight。

第一版最小实现可以是：

```text
collect_sac_gptq_stats:
  hook Linear input X
  Xq = activation_per_token_qdq_forward(X)
  accumulate Hqq = Xq^T Omega Xq
  accumulate Hqx = Xq^T Omega X

solve target:
  Bc = solve(Hqq + lambda I, (Hqx + lambda I) W^T)
  Wc = Bc^T

quantize:
  run GPTQ on target Wc with Hessian Hqq + lambda I
  use original W row scale by default
```

### 8. 风险和需要验证的点

这个方案不是无风险的，主要风险有四个：

1. **补偿目标过拟合校准集**：$W_c$ 是根据 calibration activation QDQ 解出来的，calib 太小或 slot 分布偏移时可能过拟合。需要调 $\lambda$ 和补偿强度。
2. **补偿幅度过大导致 scale 变差**：如果 $W_c$ 的 absmax 明显大于 $W$，FP8 row scale 变粗，可能损害整体列。第一版应默认使用 original weight scale，并监控 $\|W_c-W\|/\|W\|$。
3. **只补偿单层 Linear 不等价于全模型最优**：后续 residual、attention、MLP 非线性会改变误差传播。因此需要同时看 layer output MSE、block output MSE、SID NLL 和最终推荐指标。
4. **teacher-forced decode calibration 成本增加**：如果收集 `s_a/s_b/s_c` 三个 slot，calibration forward 约增加 3 倍。可以先做 `prefill_last_sid_begin` 版本，再扩展到 full slot。

### 9. 关键实验设计

为了证明 SAC-GPTQ 不是普通 weighted GPTQ 的重命名，实验必须拆开 activation-aware 和 SID-aware 两个因素：


| 方法                  | activation-aware | SID-aware | tail BF16 | 推理额外开销 |
| ----------------------- | -----------------: | ----------: | ----------: | -------------: |
| GPTQ                  |               否 |        否 |        否 |           无 |
| weighted GPTQ         |               否 |        是 |        否 |           无 |
| AC-GPTQ               |               是 |        否 |        否 |           无 |
| SAC-GPTQ              |               是 |        是 |        否 |           无 |
| weighted GPTQ + tail1 |               否 |        是 |        是 |           有 |

理想结果是：

```text
SAC-GPTQ > weighted GPTQ
SAC-GPTQ > AC-GPTQ
SAC-GPTQ 接近 weighted GPTQ + tail1
SAC-GPTQ real latency 明显优于 tail1
```

如果只在 fake quant 上接近 tail1，但 real latency 不增加，那么它就能替代当前最割裂的 tail1 分支。

### 10. 当前判断

SAC-GPTQ 的数学形式是成立的：它来自一个清晰的加权 ridge least squares 目标，并且可以被转成 GPTQ 可处理的二次型。它的创新点不是简单“给 GPTQ 加权”，而是把推理时 activation QDQ 产生的输入扰动纳入 weight search，使离线 FP8 weight 同时补偿 W 和 A 的联合误差。

更重要的是，它和 OneRec 的固定 SID 生成结构有自然结合点：SID slot 决定哪些 activation quantization error 更值得补偿，而不是通过 tail1 直接绕开低精度计算。因此，它是当前最值得替代 `grad_weighted_gptq_fp8_w8a8_tail1` 的下一代主线候选。


## 补充方案：Tail-aware Activation-Compensated GPTQ / 2026-07-07

### 1. 来自 real tail1 实验的新判断

最新 real quant 结果显示，在 `slot_grad_weighted_gptq + full_sid_multi_target + decode A16` 的基础上继续加入 prefill `tail1`，确实能修复部分小 K 指标，但对当前最关注的 K=32 指标提升已经很有限，同时带来明显时延代价：

```text
new loss no-tail1:
  avg_time_per_sample = 0.4087s
  speedup vs BF16 = 1.315x

new loss + tail1:
  avg_time_per_sample = 0.4963s
  speedup vs BF16 = 1.083x
```

这说明两点：

1. prefill 最后一个 token 的 activation 精度确实重要，tail1 仍然是有效的诊断实验；
2. 当前 slot-aware / full-SID weighted GPTQ 已经恢复了大部分 tail1 原本负责补偿的误差，继续在线保留 tail1 的性价比下降。

因此主方法不应继续采用 runtime tail1 分支。更合理的目标是：**保留 tail1 暴露出的关键 token 精度需求，但把补偿转移到离线 GPTQ 搜索中，保持 no-tail1 的推理路径和时延。**

### 2. tail1 为什么慢

当前 real tail1 不是只保护最终 logits 前的一个 token，而是在每个 `RealFP8Linear` 内部把 prefill 最后 token 单独切出，走 W8A16 路径：

```text
x_main = x[..., :-1, :]
x_tail = x[..., -1:, :]

y_main = W8A8(x_main)
y_tail = W8A16(x_tail)
y = cat(y_main, y_tail)
```

即使 q/k/v 和 gate/up 已经做了 shared-input 合并，tail1 仍会在各层引入额外 BF16 Linear、切分和拼接。因此它的时延损失是结构性的，不能指望通过简单工程优化完全消除。

### 3. Tail-aware AC-GPTQ 的核心目标

tail1 的本质不是证明最后 token 必须在线保持 BF16，而是证明：

```text
prefill last token 的 activation QDQ 误差对 SID generation 很敏感。
```

普通 GPTQ 的离线目标是：

$$
\min_{\hat W}
\left\|XW^T - X\hat W^T\right\|_{\Omega}^{2}
$$

但真实 W8A8 runtime 中 Linear 输入是 activation QDQ 后的：

$$
X_q = Q_a(X)
$$

因此 no-tail1 实际计算的是：

$$
X_q\hat W^T
$$

Tail-aware AC-GPTQ 直接把这一点写进 GPTQ 目标：

$$
\min_{\hat W}
\left\|XW^T - X_q\hat W^T\right\|_{\Omega}^{2}
$$

其中 $\Omega$ 继续承接当前 slot-aware token 权重，并额外强调 prefill last token：

$$
\omega_i =
\begin{cases}
\gamma_{\mathrm{tail}}, & i \in \text{prefill-last-token}, \\
\beta_{\mathrm{slot}}(i), & \text{otherwise}.
\end{cases}
$$

这样做的含义是：推理时仍然让最后 token 走 FP8 activation quantization，但离线搜索出来的 weight 已经主动补偿了该位置的 activation QDQ 误差。

### 4. 接入 GPTQ 的形式

先求连续补偿权重：

$$
B_c =
\left(X_q^T\Omega X_q + \lambda I\right)^{-1}
\left(X_q^T\Omega X + \lambda I\right)W^T
$$

令：

$$
W_c = B_c^T
$$

然后复用 GPTQ 主循环，只把输入从普通 weighted GPTQ 的：

```text
target_weight = W
hessian = X^T Omega X
```

替换为：

```text
target_weight = W_c
hessian = X_q^T Omega X_q + lambda I
scale_source = original W
```

`scale_source` 第一版仍建议使用原始 $W$，保持当前 fake/real GPTQ 的 FP8 scale 语义一致，避免补偿权重 $W_c$ 的少数 outlier 扩大 row scale。

### 5. 第一版最小实验

第一版不建议直接做完整 teacher-forced `s_a/s_b/s_c` decode compensation，而是先验证 prefill last-token 目标是否能替代 runtime tail1：

```text
mode:
  tail_ac_gptq_fp8_w8a8

calibration:
  ad_calib, 1024 samples
  prompt prefill rows
  Xq = current dynamic FP8 activation QDQ(X)

Omega:
  current full-SID slot-grad token weights
  + prefill last-token gamma_tail

hyperparameters:
  gamma_tail in {2, 4, 8}
  lambda in {0.01, 0.1} * mean(diag(Hqq))

runtime:
  decode A16
  no prefill tail1
```

对比实验：

| 方法 | activation-aware | tail-aware | runtime tail1 | 推理额外开销 |
| --- | ---: | ---: | ---: | ---: |
| GPTQ | 否 | 否 | 否 | 无 |
| slot-grad GPTQ full-SID | 否 | 间接 | 否 | 无 |
| slot-grad GPTQ full-SID + tail1 | 否 | 是 | 是 | 有 |
| Tail-aware AC-GPTQ | 是 | 是 | 否 | 无 |

验证指标：

1. final metrics：`pass@32`、`recall@32`、`pid_pass@32`；
2. latency：应接近 no-tail1，而不是 tail1；
3. prefill last-token Linear/block MSE：应低于 no-tail1；
4. weight drift：监控 $\|W_c-W\|/\|W\|$，避免补偿过强；
5. scale stability：监控 $W_c$ 的 absmax 是否显著大于原始 $W$。

### 6. 方法定位

Tail-aware AC-GPTQ 可以看作 SAC-GPTQ 的更小、更聚焦版本：

```text
SAC-GPTQ:
  泛化目标，补偿所有 SID role / SID slot 的 activation QDQ 误差。

Tail-aware AC-GPTQ:
  从 tail1 诊断结果出发，优先补偿 prefill last token 的 activation QDQ 误差。
```

它的论文叙事也更直接：

> tail1 作为诊断实验发现 prefill last token 对 SID generation 高敏感；我们不在在线推理阶段保留 BF16 fallback，而是在离线 GPTQ 中引入 tail-aware activation-compensated objective，使 FP8 weight 主动吸收关键 token 的 activation QDQ 误差，从而获得接近 tail1 的精度和 no-tail1 的时延。

如果第一版有效，再把 teacher-forced `s_a/s_b/s_c` rows 纳入 $\Omega$，扩展为完整 SAC-GPTQ；如果第一版无效，则说明单层 weight compensation 很难替代 runtime activation protection，应继续保留 tail1 作为 upper-bound ablation，而不是主方法。

## 新方案：CAMP-GPTQ / Catalog-Aware Margin-Preserving GPTQ / 2026-06-24

### 1. 方案定位

CAMP 的核心目标不是继续把 OneRec 类比成 MLLM，而是回到推荐任务本身：OneRec 虽然通过 SID token 自回归生成 item，但最终评测对象是封闭 item catalog 上的排序结果。推荐任务关心的不是普通 next-token distribution 是否整体接近，而是：

```text
对同一个用户，teacher 认为 item_i 应该排在 item_j 前面，量化后这个相对顺序是否还能保持。
```

这和 MLLM 的 token/modality heterogeneity 不同。推荐模型有三个更强的任务结构：

- **封闭候选集**：最终 item 来自 catalog，而不是开放词表自由生成；
- **SID 路径分数**：一个 item 的分数由 `s_a/s_b/s_c` 三段条件概率相加得到；
- **同前缀 hard negative**：共享 `s_a` 或 `s_a/s_b` 的 item 往往语义相近，量化扰动更容易改变它们的相对排序。

因此，CAMP-GPTQ 的主张是：**把 catalog item ranking margin 变成 GPTQ 的校准信号**。它不直接优化不可导的 top-k overlap，也不再只按 token 形式加权，而是用推荐 catalog 中的 item pair margin 来决定哪些校准激活方向更值得保护。

### 2. 量化方法承接：第一版接 GPTQ Hessian

CAMP 需要一个稳定、可落地、推理零开销的量化承接方式。第一版建议明确选择 **GPTQ Hessian weighting**，而不是 SmoothQuant、LET/LWC 或 learnable block reconstruction。

理由如下：

1. 当前最有效的 active path 已经是 GPTQ / weighted GPTQ，代码里 `collect_gptq_hessians` 已支持 token weights，接入成本最低。
2. catalog margin 产生的是“哪些 token row 的局部误差会改变 item 排序”的重要性，天然可以转成 GPTQ 的 row-weighted Hessian。
3. GPTQ 是离线 weight search，最终仍然导出普通 FP8 weight，不增加 real quant 推理开销。
4. 旧的 `ranking_margin + smoothquant` 已经在历史 notes 中表现不好，说明 margin 信号不适合简单塞进 SmoothQuant scale；GPTQ 的二阶误差补偿比 smoothing 更适合承接这种任务敏感信号。·

因此第一版主方法定义为：

```text
CAMP-GPTQ = catalog item margin sensitivity -> token row weights -> GPTQ Hessian -> FP8 weight
```

activation 侧暂时保持当前 per-token dynamic FP8 QDQ。也就是说，CAMP-GPTQ 不是 activation quantizer 设计，而是一个推荐任务感知的 weight PTQ 方法。后续如果要同时处理 W/A 割裂，可以把 CAMP 权重作为 Omega 接入 SAC-GPTQ，形成 CAMP-SAC-GPTQ，但不建议第一版就把两个变量混在一起。

### 3. Catalog item 路径分数

对校准用户 `u` 和 catalog item `i`，设 item 的 SID 为：

```text
sid(i) = (a_i, b_i, c_i)
```

用 BF16 teacher 做 teacher-forcing，定义 item 路径分数：

$$
S_T(u,i) =
log p_T(a_i | u)
+ log p_T(b_i | u, a_i)
+ log p_T(c_i | u, a_i, b_i)

$$

量化模型对应：

$$
S_Q(u,i) =
log p_Q(a_i | u)
+ log p_Q(b_i | u, a_i)
+ log p_Q(c_i | u, a_i, b_i)

$$

对于 teacher 排序中 `i` 高于 `j` 的 item pair，定义 teacher margin：

$$
M_T(u,i,j) = S_T(u,i) - S_T(u,j)

$$

CAMP 不直接用 top-k set overlap，而是关注这个连续 margin。margin 小的 pair、同 SID 前缀的 pair、teacher top boundary 附近的 pair 更容易因量化扰动翻转，因此应给更高权重。

### 4. 候选集构造

每个校准用户构造一个小候选集 `C_u`，不需要遍历全 catalog：

```text
C_u = teacher top-B items
    + ground-truth item, optional
    + same-a hard negatives
    + same-a-b hard negatives
    + popular/random negatives
```

推荐第一版配置：

```text
teacher top-B: 32
same-a negatives: 8
same-a-b negatives: 8
random/popular negatives: 8
max pairs per user: 8
```

pair 选择按 teacher 分数排序后进行：

- 优先选 top boundary pair，例如 rank 16 vs rank 17、rank 32 vs rank 33；
- 优先选同 `s_a` 或同 `s_a/s_b` 前缀的 hard negative pair；
- 只使用 teacher margin 为正的 pair，避免 teacher 自身排序不确定；
- 对 margin 极大的 easy pair 降权或丢弃。

pair 权重可以用一个简单的 boundary-aware 函数：

$$
rho(M_T, type) = clip(exp(-M_T / T), rho_min, rho_max) * gamma(type)

$$

其中 `type` 表示 pair 类型，例如：

```text
same-a-b: gamma high
same-a: gamma medium
top-boundary: gamma medium
random: gamma low
```

### 5. 从 catalog margin 到 GPTQ Hessian

这是 CAMP-GPTQ 的关键承接点。

普通 GPTQ 对某个 Linear 使用校准输入 `X` 构造 Hessian：

$$
H = X^T X

$$

weighted GPTQ 使用 token 权重：

$$
H = X^T diag(w) X

$$

CAMP-GPTQ 的区别是：`w` 不来自 token 形式，也不来自普通 SID CE，而是来自 catalog item margin 的敏感性。

对第 `l` 层，选定一个 item pair `(i, j)`，在 BF16 teacher 上反传 teacher margin：

$$
M_T(u,i,j) = S_T(u,i) - S_T(u,j)

$$

得到该 margin 对第 `l` 层 hidden row 的梯度：

$$
g_{l,r}^{u,i,j} = dM_T(u,i,j) / dh_{l,r}

$$

其中 `r` 是展平后的 token row。用梯度范数估计该 row 的排序敏感性：

$$
alpha_{l,r} = Normalize(
  eps + mean_{(u,i,j)} rho(M_T,type) * ||g_{l,r}^{u,i,j}||_2^2
)

$$

然后对该层内每个 Linear，用同一组 row weights 构造 CAMP Hessian：

$$
H_m^{CAMP} =
( sum_r alpha_{l,r} x_{m,r} x_{m,r}^T ) / ( sum_r alpha_{l,r} )

$$

这里 `x_{m,r}` 是第 `m` 个 Linear 的输入 row。直观上，CAMP-GPTQ 优化的是：

```text
如果某个 token row 的 hidden 扰动会显著改变 catalog item margin，
那么这个 row 对 GPTQ Hessian 的贡献就更大，
weight quantization 会更倾向保护这个输入方向。
```

这可以看成 catalog ranking margin 下的 diagonal Fisher approximation。它比直接优化 top-k overlap 更稳，因为整个信号是连续的 teacher margin gradient；它也比 SID CE gradient 更推荐专属，因为梯度目标是 item pair ranking，而不是单个 SID token 的分类损失。

### 6. 算法流程

第一版 CAMP-GPTQ 的完整 pipeline：

```text
Input:
  BF16 OneRec teacher
  calibration users
  item catalog with SID mapping
  FP8 weight quantizer
  dynamic activation fake quantizer

Stage A: candidate construction
  for each calibration user u:
    run teacher beam search to get top-B generated items
    add ground-truth item if available
    sample same-a and same-a-b hard negatives from catalog
    sample small random/popular negatives

Stage B: teacher path scoring
  for each candidate item i in C_u:
    teacher-force sid(i) = (a_i, b_i, c_i)
    compute S_T(u, i)
  build hard item pairs P_u by teacher margin and prefix type

Stage C: margin sensitivity collection
  for each selected layer l:
    for selected pairs (u, i, j):
      backward M_T(u,i,j) on BF16 teacher
      collect row sensitivity ||dM_T / dh_l||^2
    normalize and clip alpha_l to mean 1

Stage D: CAMP-GPTQ Hessian collection
  for each layer l:
    run calibration forward
    pass alpha_l as token_weight_batches
    collect H_m^CAMP for every Linear m in the layer

Stage E: GPTQ quantization
  for every Linear m:
    run existing FP8 GPTQ with H_m^CAMP
    replace Linear with GPTQFakeQuantLinear

Inference:
  same as current W8A8 fake/real quant path
  activation remains per-token dynamic FP8 QDQ
  no extra online branch
```

### 7. 与现有模式的关系

推荐新增 mode：

```text
camp_gptq_fp8_w8a8
camp_gptq_fp8_w8a8_tail1
```

后续如果 SAC-GPTQ 先跑通，可以再加入：

```text
camp_sac_gptq_fp8_w8a8
```

三者关系：


| mode                          |           ranking-aware | activation-aware weight target | tail BF16 | 在线额外开销 |
| ------------------------------- | ------------------------: | -------------------------------: | ----------: | -------------: |
| `grad_weighted_gptq_fp8_w8a8` |   弱，SID CE/token role |                             否 |        否 |           无 |
| `camp_gptq_fp8_w8a8`          | 是，catalog item margin |                             否 |        否 |           无 |
| `camp_gptq_fp8_w8a8_tail1`    | 是，catalog item margin |                             否 |        是 |           有 |
| `camp_sac_gptq_fp8_w8a8`      | 是，catalog item margin |                             是 |        否 |           无 |

第一版建议只实现 `camp_gptq_fp8_w8a8`。如果它能优于 `grad_weighted_gptq_fp8_w8a8`，说明推荐 ranking margin 信号是有效的；如果还不能替代 tail1，再尝试 `camp_sac_gptq_fp8_w8a8` 或 `camp_gptq_fp8_w8a8_tail1`。

### 8. 为什么不是直接优化 margin loss

不要第一版就做 learnable block reconstruction 或直接优化 pairwise margin loss，原因是：

- 会引入可学习参数或 STE，容易从 PTQ 滑向小规模 QAT；
- pairwise margin loss 需要 student forward/backward，计算成本高；
- 小 calibration set 下容易过拟合具体 item pair；
- 当前代码已经有 GPTQ Hessian 收集和量化主路径，先把 margin 信号转成 Hessian 权重更稳。

因此 CAMP-GPTQ 的方法边界是：

```text
margin 只用于离线估计校准重要性；
真正执行量化的仍是 GPTQ；
推理图不加入 margin module、adapter 或 reranker。
```

### 9. 实验设计

必须拆清楚三件事：普通二阶误差补偿、SID token 信号、catalog ranking 信号。

推荐 ablation：


| 方法                    | 目的                                                           |
| ------------------------- | ---------------------------------------------------------------- |
| naive W8A8              | 基础低精度损失                                                 |
| GPTQ W8A8               | 普通二阶 weight PTQ                                            |
| weighted GPTQ           | 手工 token role 是否有效                                       |
| grad weighted GPTQ      | SID CE gradient 是否有效                                       |
| CAMP-GPTQ               | catalog margin signal 是否有效                                 |
| CAMP-GPTQ + tail1       | margin signal 和关键 activation 保护是否互补                   |
| CAMP-SAC-GPTQ, optional | margin signal 和 activation-compensated weight target 是否互补 |

除最终 `pass@k/recall@k/pid_recall@k` 外，需要新增诊断指标：

```text
item score MSE:
  mean_i |S_Q(u,i) - S_T(u,i)|^2

pair margin MSE:
  mean_{i,j} |M_Q(u,i,j) - M_T(u,i,j)|^2

pair flip rate:
  percentage of teacher-positive pairs where M_Q < 0

same-prefix flip rate:
  pair flip rate only on same-a and same-a-b negatives

boundary flip rate:
  pair flip rate around top-16/top-32 boundary
```

如果 CAMP-GPTQ 是有效的，应该观察到：

```text
CAMP-GPTQ 的 pair margin MSE 低于 grad weighted GPTQ
CAMP-GPTQ 的 same-prefix flip rate 更低
CAMP-GPTQ 的 boundary flip rate 更低
CAMP-GPTQ 在 recall@32 / pid_recall@32 上更稳定
```

### 10. 实现注意点

1. **不要用 ground-truth label 强行训练排序**。第一版优先使用 BF16 teacher ranking，ground-truth item 只作为候选补充，pair 顺序仍由 teacher score 决定。这样保持 PTQ distillation 属性，避免校准集标签过拟合。
2. **pair 数必须小**。建议每个 user 最多 4-8 个 pair，否则 backward 成本很高。
3. **alpha 要做归一化和裁剪**。建议每层 `alpha` mean normalize 到 1，并做 `[0.1, 10]` 或更保守范围裁剪，避免少数 pair 主导 Hessian。
4. **先做 layer-wise row weight，不做 module-wise 细粒度**。同一层内 q/k/v/o/gate/up/down 使用相同 token row weight，保持稳定和实现简单。
5. **先不改 activation quantizer**。CAMP 的目标是验证推荐 ranking signal 是否比 SID token weighting 更有效；activation 侧保持 current dynamic QDQ。

### 11. 当前判断

CAMP-GPTQ 比当前 token/gradient weighted GPTQ 更推荐专属，因为它的基本对象不再是“token role”，而是 catalog item pair 的排序 margin。这个信号很难直接迁移到普通 MLLM，因为它依赖：封闭 item catalog、SID 到 item 的映射、同前缀 hard negative、top-k 推荐边界。

同时，CAMP-GPTQ 又足够可落地：它不要求新 kernel，不要求在线 rerank，不要求 learnable quant parameter，第一版只需要把 catalog margin sensitivity 转成 GPTQ Hessian 权重。它适合作为下一条主线，用来回答当前最核心的问题：**推荐大模型量化是否应该保持 item ranking margin，而不是只保持 token-level reconstruction。**

## CAMP 的量化承接器选择：为什么第一版使用 GPTQ / 2026-06-24

### 1. 核心判断

CAMP 不应被理解成“必须绑定 GPTQ 的一个 trick”。更准确地说：

```text
CAMP 是推荐任务信号：
  catalog item pair margin sensitivity

GPTQ 是第一版量化承接器：
  row-weighted Hessian -> weight quantization
```

选择 GPTQ 不是因为 GPTQ 最新或理论上唯一正确，而是因为 CAMP 产生的信号形态和 GPTQ 的输入接口最匹配。CAMP 估计的是第 `r` 个 calibration row 对 catalog item margin 的敏感性：

$$
\alpha_r = Sensitivity(row_r, catalog\ margin)

$$

而 GPTQ 的 Hessian 本身就是 calibration row 的二阶累积：

$$
H = \sum_r x_r x_r^T

$$

因此 CAMP 可以直接把推荐排序信号写成：

$$
H_{CAMP} = \sum_r \alpha_r x_r x_r^T

$$

这个接口非常干净：catalog margin 负责产生 row importance，GPTQ 负责在该 Hessian 下做二阶 weight rounding。推理阶段仍然导出普通 FP8 weight，不增加在线分支。

### 2. 和其他量化方法的比较


| 方法族                        | 代表方法                 |               是否适合第一版 CAMP | 判断                                                                                                                                                                       |
| ------------------------------- | -------------------------- | ----------------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPTQ / OPTQ                   | GPTQ                     |                                高 | CAMP row sensitivity 可以直接进入 weighted Hessian，数学接口最自然，实现也最贴近当前代码。                                                                                 |
| AWQ                           | AWQ                      |                                中 | AWQ 使用 activation 统计识别 salient weight channel，本质是 channel scaling。CAMP 的信号是 row/item-pair sensitivity，若压缩成 channel importance 会损失 pair-level 信息。 |
| SmoothQuant / SQ+             | SmoothQuant              |                                低 | SmoothQuant 解决 activation outlier，把难度迁移到 weight。历史`ranking_margin + smoothquant` 已经效果不好，说明 margin 信号不适合简单塞进 smoothing scale。                |
| OmniQuant / LET / LWC         | OmniQuant                |                                中 | 可以把 CAMP margin loss 加入 block reconstruction，但会变成 learnable PTQ，校准成本和过拟合风险高于 GPTQ Hessian weighting。                                               |
| 旋转 / Hadamard / incoherence | QuaRot, SpinQuant, QuIP# | 中高，但更适合作为 preconditioner | 它们解决 outlier、heavy-tail、坐标轴不均匀和 incoherence 问题，不直接承接 catalog ranking signal。                                                                         |
| Affine / Flat transformation  | AffineQuant, FlatQuant   |                      有潜力但复杂 | 可以学习等价 affine transform，但若用 CAMP margin 优化 transform，会变成更重的 learnable PTQ，还需要考虑 transform 融合和 real latency。                                   |
| 系统协同量化                  | QServe / QoQ             |                                低 | 主要解决 kernel、dequant overhead、serving throughput，不是推荐任务信号的校准承接点。                                                                                      |
| codebook / vector quant       | AQLM, QuIP# codebook     |                        当前不优先 | 更适合极低 bit weight-only 压缩；当前目标是 FP8 W8A8 和真实 FP8 GEMM。                                                                                                     |

### 3. 为什么不第一版使用旋转或 Hadamard

旋转类方法的典型形式是：

$$
XW = (XR)(R^TW)

$$

或者在 residual、attention、MLP、KV cache 中插入可以相互抵消或融合的正交变换。它们的核心价值是：

```text
把 outlier 分散掉；
让 activation/weight 分布更平；
降低低 bit 量化难度；
改善 W4A4、KV4 等极端量化。
```

但 CAMP 要解决的问题不同。CAMP 的核心问题是：

```text
哪些局部量化误差会改变 catalog item pair 的排序 margin？
```

如果直接使用固定 Hadamard 或随机旋转，catalog margin 信号没有进入优化目标；如果学习 rotation 并使用 catalog margin loss 优化：

$$
\min_R L_{catalog-margin}(Q(XR), Q(R^TW))

$$

它会变成一个重型 learnable PTQ，带来三个问题：

1. calibration 成本明显上升；
2. 小校准集下更容易过拟合少量 item pair；
3. real quant 中需要额外考虑 rotation/fused Hadamard 的在线开销。

因此，旋转/Hadamard 不适合作为 CAMP 第一版的主承接器。它更适合作为第二阶段的 **distribution preconditioner**。

### 4. 更合理的组合方式

如果后续发现 activation/channel outlier 仍然是主要瓶颈，可以把旋转接在 CAMP 前面，而不是替代 CAMP-GPTQ：

```text
CAMP-Rotate-GPTQ:
  rotation/Hadamard 负责改善分布；
  CAMP 负责提供 catalog margin row weights；
  GPTQ 负责离散 weight search。
```

形式上，令：

$$
X' = XR

$$

则 CAMP Hessian 变成：

$$
H'_{CAMP} = (XR)^T \Omega (XR)
= R^T X^T \Omega X R

$$

也就是说，rotation 改变坐标系，CAMP 决定哪些 row/pair 更重要，GPTQ 在新的坐标系中做二阶 rounding。这三者是互补关系，不是替代关系。

### 5. 推荐路线

当前建议推进顺序：

```text
Stage 1:
  CAMP-GPTQ
  验证 catalog margin signal 是否比 SID/token weighting 更有效。

Stage 2:
  CAMP-SAC-GPTQ
  在 catalog margin weighting 基础上，引入 activation-compensated weight target。

Stage 3:
  CAMP-Rotate-GPTQ 或 CAMP-Rotate-SAC-GPTQ
  如果仍观察到 activation/channel outlier 或 FP8 range 问题，再加入 rotation/Hadamard/affine preconditioner。
```

因此，第一版使用 GPTQ 的理由是：它最直接承接 CAMP 的 row-level ranking sensitivity，且能最大程度复用当前 fake/real quant 路径。旋转/Hadamard 很有价值，但它解决的是分布预处理问题，不是推荐 catalog ranking signal 的主承接问题。
