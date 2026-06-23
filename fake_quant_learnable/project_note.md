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

| run | pass@1 | pass@16 | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_w8a8_1p7b_ad_calib1024 | 0.019773 | 0.139947 | 0.204705 | 0.070265 | 0.190748 | 0.064848 | 当前主要 W8A8 对照 |
| smoothquant_w8a8_1p7b_ad_calib1024_fixed_fold_full | 0.018760 | 0.139196 | 0.202416 | 0.069607 | 0.188759 | 0.064348 | SQ fold，效果略低于 baseline |
| smoothquant_w8a8_fold_ad_calib1024_1p7b | 0.019097 | 0.139309 | 0.204893 | 0.070519 | 0.190710 | 0.065231 | SQ fold，和 baseline 基本持平 |
| smoothquant_w8a8_alpha0p4 | 0.019510 | 0.142948 | 0.203955 | 0.070333 | 0.189322 | 0.064769 | alpha=0.4，@16 略好，@32 持平 |
| m2_lwt_let_1p7b_ad_calib1024_sqinit_fixed_fold_full | 0.018572 | 0.140922 | 0.202491 | 0.070030 | 0.188984 | 0.064747 | LWT+LET，SQ 初始化，未优于 baseline |
| m2_lwt_let_sqinit_ad_calib1024_1p7b | 0.018872 | 0.140134 | 0.202229 | 0.068842 | 0.188009 | 0.063464 | 新版 SQ init，效果偏低 |
| baseline_w8a8_skip_last_ad_calib1024_1p7b_full | 0.019322 | 0.140697 | 0.205118 | 0.070617 | 0.191423 | 0.065281 | 最后一层不量化，和 baseline 接近 |
| w8a8_tail1_ad_calib1024_1p7b_full | 0.020185 | 0.145500 | 0.211533 | 0.072724 | 0.197351 | 0.067080 | tail token activation 保护，结果较好但需要复跑确认 |

### 梯度 group 权重 probe / 2026-06-15

目的：在实现 `grad_weighted_gptq_fp8_w8a8` 的 layer-wise group 版本前，先看 calib 梯度统计出的 group 权重和手工 group 权重的差异。

设置：OneRec-1.7B，calib 前 32 条，全 28 层；梯度目标沿用当前 v2，即 `last prompt position -> first ground-truth s_a` 的 CE loss；token sensitivity 为 `mean(|hidden * grad|)`，再使用当前 gradient config 做 clip / floor / mean normalize，最后按 prompt role group 聚合。

手工权重 raw 为 `text=10, history_sid=1, interest_sid=5, sid_boundary=2`。由于 GPTQ Hessian 收集前会按样本均值归一化，probe 中手工 group 的 normalized 均值如下：

| group | manual normalized |
|---|---:|
| text | 3.5205 |
| history_sid | 0.3548 |
| interest_sid | 1.6937 |
| sid_boundary | 0.7018 |

逐层梯度 group 权重：

| layer | text | history_sid | interest_sid | sid_boundary |
|---:|---:|---:|---:|---:|
| manual | 3.5205 | 0.3548 | 1.6937 | 0.7018 |
| 0 | 2.3065 | 0.8292 | 1.9489 | 0.4468 |
| 1 | 2.7725 | 0.7827 | 1.9727 | 0.3470 |
| 2 | 2.8290 | 0.7375 | 1.9075 | 0.4044 |
| 3 | 2.9290 | 0.7132 | 1.8845 | 0.4094 |
| 4 | 2.9798 | 0.6946 | 1.8580 | 0.4244 |
| 5 | 3.1133 | 0.6678 | 1.8124 | 0.4301 |
| 6 | 3.2066 | 0.6422 | 1.7805 | 0.4419 |
| 7 | 3.2373 | 0.6258 | 1.7633 | 0.4572 |
| 8 | 3.2671 | 0.6048 | 1.7324 | 0.4829 |
| 9 | 3.3374 | 0.5702 | 1.7123 | 0.5077 |
| 10 | 3.4621 | 0.5054 | 1.6778 | 0.5551 |
| 11 | 3.4867 | 0.4855 | 1.6674 | 0.5738 |
| 12 | 3.4977 | 0.4838 | 1.6727 | 0.5704 |
| 13 | 3.3820 | 0.4832 | 1.7018 | 0.5962 |
| 14 | 3.2321 | 0.4942 | 1.7700 | 0.6053 |
| 15 | 3.1795 | 0.5023 | 1.8379 | 0.5878 |
| 16 | 3.1632 | 0.4928 | 1.8752 | 0.5900 |
| 17 | 3.1142 | 0.5194 | 1.8762 | 0.5747 |
| 18 | 3.1354 | 0.5289 | 1.9471 | 0.5317 |
| 19 | 3.1066 | 0.5393 | 1.9286 | 0.5356 |
| 20 | 2.9439 | 0.5701 | 1.9664 | 0.5371 |
| 21 | 2.9470 | 0.5453 | 1.9674 | 0.5639 |
| 22 | 3.1310 | 0.5596 | 2.1680 | 0.4183 |
| 23 | 2.6226 | 0.6035 | 2.3082 | 0.4747 |
| 24 | 2.5130 | 0.6440 | 2.4645 | 0.4060 |
| 25 | 2.3945 | 0.6588 | 2.3881 | 0.4534 |
| 26 | 2.2611 | 0.6731 | 2.2838 | 0.5162 |
| 27 | 3.2884 | 0.5272 | 1.1557 | 0.7731 |

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

| prompt role | raw weight |
|---|---:|
| text | 10.0 |
| history_sid | 1.0 |
| interest_sid | 5.0 |
| sid_boundary | 2.0 |

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

| 参数 | 默认值 | 作用 |
|---|---:|---|
| clip_percentile | 99.0 | 对有效 token 的 sensitivity 做 99 分位截断，避免极端 token 主导 Hessian |
| weight_floor | 0.05 | 给有效 token 设置最小权重，避免某些 token 完全消失 |
| normalize_mean | True | 有效 token 均值归一化到 1 |
| eps | 1e-12 | 数值稳定 |

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

