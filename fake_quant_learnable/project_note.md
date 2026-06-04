# fake_quant_learnable Project Note

更新时间：2026-06-02

`fake_quant_learnable` 是当前 OneRec 推荐大模型 PTQ 量化实验的主要工作目录。核心任务是：在 OneRec AD item prediction 任务上，研究 W8A8 fake quant、SmoothQuant、learnable weight clipping LWT、learnable equivalent transformation LET，以及后续更贴近推荐目标的量化方法。

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

这是所有 learnable / SmoothQuant 实验的主要对照组。

### smoothquant_w8a8

先用 calibration prompt 收集每个 Linear 输入激活 absmax，再按 SmoothQuant 公式计算 scale，对 q/k/v、o_proj、gate/up/down 等 OmniQuant 风格位置做平滑。当前实现支持 fold：能等价折叠的位置会把 scale 合并进 LayerNorm 或相邻 Linear，量化时输入分布和权重分布更接近实际部署设置。

当前默认 `alpha=0.4`，这是为了观察 alpha 调整后的趋势。已有结果显示 SmoothQuant 确实能显著改变 activation/weight 数值分布，但最终推荐指标提升不稳定、幅度很小。

### m1_lwt

M1 只学习 weight clipping threshold。每个 Linear 的输出通道有可学习 clipping multiplier，训练目标是当前 block 输出和 BF16 teacher block 输出的 MSE。

M1 的训练 loss 通常能下降，但推荐指标不一定提升。当前理解是：OneRec 权重本身较平滑，单独调整截断阈值的收益很低，MSE 的小幅下降不一定对应 SID 排名改善。

### m2_let / m2_lwt_let

M2 加入 LET scale。LET 对 Linear 输入做 `x / s`，对权重做 `W * s`，数学上在不量化时等价；量化后希望通过重分配 activation 和 weight 的异常值来降低损失。

- `m2_let`：只训练 LET，LWT 固定。
- `m2_lwt_let`：同时训练 LET 和 LWT。
- `LET_INIT=smoothquant`：用 SmoothQuant scale 初始化 LET。

目前已经对齐了 LWT/LET 分学习率、按 epoch 训练、SmoothQuant 初始化和 fold 等逻辑。整体流程和 OmniQuant 的 block-wise reconstruction 思路基本一致，但当前效果仍然不明显。

## 代码结构

```text
fake_quant_learnable/
  quant.py                  # FP8 E4M3 QDQ 和 STE proxy
  modules.py                # Baseline/Learnable/Frozen Linear wrappers
  apply.py                  # Linear 替换、shared input QDQ、learned params 导出/加载
  calibrate_m1_lwt.py       # block output MSE 校准循环，训练 LWT/LET 参数
  runtime_utils.py          # tensor tree detach/device helper
  smoothquant_runtime.py    # SmoothQuant scale 收集、fold、W8A8 包装
  tail_protect.py           # tail token activation 保护的 W8A8 变体
  run_m1_onerec_ad.py       # 主 runner：baseline/SQ/M1/M2 训练、生成、评测
  run_learnable_quant_ad.sh # 统一 shell 入口，固定大部分默认参数
  run_w8a8_tail_protect_ad.py # tail-token 保护实验入口
  inspect_smoothquant_distribution.py # SQ 前后 activation/weight 分布探针
  tests/                    # toy model 单测和 runner 单测
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
MODE=smoothquant_w8a8 DEVICE=cuda:0 RUN_NAME=smoothquant_w8a8_alpha0p4 \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

M2 LWT+LET，SmoothQuant 初始化：

```bash
MODE=m2_lwt_let LET_INIT=smoothquant DEVICE=cuda:0 RUN_NAME=m2_lwt_let_sqinit_ad_calib1024_1p7b \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

只校准一层，快速看收敛：

```bash
MODE=m2_lwt_let LET_INIT=smoothquant LAYERS=0 CALIB_ONLY=1 EPOCHS=1 DEVICE=cuda:6 \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
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

### 历史 AD1000 子集

| run | pass@1 | pass@16 | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_w8a8_ad1000_calib_offset_1000 | 0.019000 | 0.164000 | 0.233000 | 0.081563 | 0.220000 | 0.076784 | 旧子集 baseline |
| m1_lwt_ad1000_calib_offset_1000 | 0.020000 | 0.153000 | 0.216000 | 0.078157 | 0.207000 | 0.074029 | M1 比 baseline 差 |
| m2_lwt_let_ad1000_calib_offset_1000 | 0.024000 | 0.157000 | 0.224000 | 0.082351 | 0.212000 | 0.078189 | 曾经略有 recall@32 收益，但 pass@32 仍低 |
| smoothquant_w8a8_1p7b_olddata_ad1000_calib128_offset1000_fixed | 0.019000 | 0.173000 | 0.243000 | 0.085479 | 0.226000 | 0.080742 | 旧数据/旧配置下 SQ 较好 |

## 当前结论

1. W8A8 baseline 和 SmoothQuant W8A8 在全量 test 上基本区分不开。SQ 能压制 activation 异常值，也能通过 fold 改变权重/激活分布，但推荐指标提升很小。

2. M1 的 block MSE 可以下降，但推荐指标可能下降。这说明纯 block output MSE 不是 OneRec SID ranking 的充分代理目标。

3. M2 LWT+LET 的逐层 MSE 优化幅度很小，部分层甚至 final loss 高于 initial loss。当前更像是在初始点附近轻微扰动，难以稳定改变最终 SID 排名。

4. 保护最后一层全精度没有明显收益，说明“只保护高层整层”不是主要突破口。tail-token activation 保护有一次全量结果看起来更好，但需要复跑和更细粒度 ablation 判断是否稳定。

5. 激活分布可视化显示，高层存在明显 channel-wise 异常值，并且某些异常 channel 在历史 SID session 区间有结构化差异；token-wise 的整体异常不如 channel-wise 明显。

6. beam search 仍然基于全 vocab logits 生成，而不是只限制 SID logits。OneRec 还有文本性推荐理解任务，因此直接限制 vocab 空间不适合作为通用推理实现。

7. 新增的 `sid_tf_nll` / `sid_tf_ppl` 可以作为辅助诊断指标：如果 W8A8 和 SQ 的 pass/recall 区分不开，可以观察真实 SID token 的 teacher-forcing likelihood 是否有差异。

## 已知风险和注意事项

- 当前 `results/` 里混有历史实验、失败实验、subset 实验和全量实验，不能简单按目录名下结论。
- custom beam / staged prefill-decode 实验已被判定和 HF generate 不可直接比较，相关结果只用于排查，不应作为论文主结果。
- 服务器当前 namespace 配置可能导致 Codex sandbox 命令失败，必要时需要提权执行读写命令。
- batch 推理在当前实现和 OpenOneRec 风格下不一定更快；单条 HF generate 目前更接近稳定设置。
- 如果开启 `COMPUTE_SID_PPL=1`，每条样本会额外做 teacher-forcing forward，评测耗时会增加。

## 后续研究方向

### 1. 用 SID teacher-forcing 指标诊断 SQ 是否真的无效

先对 W8A8 baseline、SmoothQuant W8A8、M2 LWT+LET 跑同样的 `COMPUTE_SID_PPL=1`，比较：

- `sid_tf_nll`
- `sid_tf_ppl`
- pass@k / recall@k

如果 SQ 能降低 SID NLL 但不提升 recall，说明 ranking/top-k 机制和 likelihood 仍有 gap；如果 SID NLL 也没有降低，则 SQ 对推荐输出本身确实帮助有限。

### 2. 从推荐目标反推量化目标

当前普通 block MSE 太弱。更合理的方向是围绕 SID logits / SID margin / top-k ranking 设计目标，例如：

- teacher-forcing SID token NLL 加权的 layer/block reconstruction；
- 对候选 SID logits 的 margin 保真；
- top-k SID 路径 ranking stability；
- 对影响 SID margin 的 channel/group 做混合精度或保护。

难点是 SID 决策信号跨层分布，不一定集中在某个显式 channel，需要用梯度或敏感度来定位。

### 3. 继续研究 activation 异常值结构

已有可视化显示：高层异常 channel 更明显，并且历史 SID session 区间有结构化差异。可以继续沿着以下问题看：

- 哪些 channel 对 SID logits margin 更敏感；
- session 区间的锯齿状模式是否对应 `<s_a_*>/<s_b_*>/<s_c_*>` token 类型；
- 异常 channel 是否可以通过 token-aware 或 SID-aware scaling 单独处理。

### 4. 复核 tail-token activation 保护

`w8a8_tail1` 当前全量结果好于 baseline，但需要严格复跑：

- 同一数据、同一模型、同一 runner；
- tail_tokens=0/1/2/3 ablation；
- 只保护 prefill 最后 token vs decode token；
- 是否影响 total time 和显存。

如果稳定，这可能比 SmoothQuant/LWT/LET 更贴近推荐生成机制。

### 5. 考虑推荐专用 PTQ，而不是继续只做平滑

SmoothQuant、LET、LWT 的本质仍是降低量化误差或平滑异常值；当前证据显示它们上限可能不高。后续可以把重点转向推荐任务特有的信息：SID token、历史 session、候选 item margin、beam path 稳定性、PID 映射后的排序稳定性等。

### 6. beam-path stability PTQ

  推荐最终看 top-k beams，所以可以直接让量化模型复现 FP 模型的 beam expansion：

  step 1: preserve top SID-a candidates
  step 2: preserve each prefix 下的 SID-b ranking
  step 3: preserve SID-c ranking / item mapping

  这比 PPL 更贴近 pass@32/recall@32。SAPO 最近也强调 SID 生成里 exact-match outcome reward 很稀疏，应该按 reasoning step / SID token 做 credit assignment，而不是把整个序列混在一起。(SAPO
  (https://arxiv.org/abs/2605.17648)) 这个思想可以迁移到量化：按 SID step 分配量化误差预算。

### 7. SID-position-aware precision allocation

  SID 是层级语义 ID。TIGER 里也强调 Semantic ID 是由多个 codeword 组成，相似 item 会共享前缀，早期 token 更像 coarse semantic routing，后续 token refine item。(TIGER
  (https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf))

  所以 s_a 错了可能直接走到错误大类，s_b/s_c 错了影响更细。可以做：

  s_a 位置更高精度 / 更强校准权重
  s_b 次之
  s_c 再次

  这就是推荐任务特有的“token position importance”，不是普通 LLM 的均匀 next-token objective。

### 8. SID-gradient channel protection

  AWQ 的核心观察是：不是所有 weight 都同等重要，重要 channel 应该从 activation 分布判断。(AWQ (https://arxiv.org/abs/2306.00978)) 你可以把它改成推荐版本：

  不是 activation 大的 channel 重要
  而是对 SID margin / beam ranking 敏感的 channel 重要

  用 teacher-forcing SID loss 或 SID margin loss 反传，得到每层 channel saliency，然后：

  - top channel 不量化 activation；
  - 或 top channel 用更小 clipping；
  - 或 top group 用 W8A16 / higher precision；
  - 或对这些 channel 做更激进的 scale equalization。

  这个方向比“平滑异常值”更有推荐味道。