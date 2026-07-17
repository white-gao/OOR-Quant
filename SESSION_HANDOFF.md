# OOR-Quant 会话迁移说明

本文用于新会话快速接手仓库。内容以当前代码和已完成实验为准；不要仅根据旧命令或目录名猜测方法配置。

## 1. 研究目标

本项目研究 **LLM-based generative recommendation 的 FP8 post-training quantization (PTQ)**。

- 基座模型：OpenOneRec / OneRec-1.7B，主要任务域为 `ad`。
- 推荐输出：一个 item 由三段 Semantic ID (SID) 组成，即 `SID-a -> SID-b -> SID-c`。
- 总目标：在尽量保持推荐指标的条件下，将主要 Linear 计算置于真实 FP8 路径，获得端到端推理加速。
- 论文层面的核心问题：通用 LLM PTQ 将 token 近似视作同质校准样本；OneRec 中 text、SID、boundary 及不同 SID 位置具有结构差异。需要找出 token 异质性在量化误差中的**可验证、可转化为算法**的影响。

当前 FP8 类型为 `torch.float8_e4m3fn`。权重使用 per-output-channel scale，activation 默认使用运行时 per-token dynamic scale。

## 2. 目录结构

```text
OOR-Quant/
├── fake_quant/                         # 早期 fake quant，仅作历史参考
├── fake_quant_learnable/               # 算法研究、fake quant、GPTQ/GPTAQ、probes
│   ├── gptq.py                          # GPTQ、GPTAQ、conditional GPTQ 核心
│   ├── quant.py                         # FP8 QDQ 参考实现
│   ├── apply.py / modules.py            # fake quant 模块替换和运行时
│   ├── token_weights.py                 # token group/梯度权重工具
│   ├── probe_slot_*.py                  # embedding、activation、outlier channel probes
│   ├── project_note.md                  # 长期技术笔记、调研与历史判断
│   └── results/analysis/                # probe 图、csv、pt
├── real_quant/                          # 真实 FP8 runtime 和时延/精度实验
│   ├── full_precision/                  # BF16 baseline runner
│   └── naive_w8a8/
│       ├── modules.py                   # RealFP8Linear、FP8 QDQ、scaled_mm
│       ├── apply.py                     # Linear 替换与 QKV/gate-up shared-input wrapper
│       ├── gptq_runtime.py              # layer streaming、Hessian/GPTAQ 校准
│       ├── run_hf_naive_w8a8.py         # 普通 real quant CLI 入口
│       ├── conditional_gptq.py          # conditional GPTQ
│       ├── stage_rescue.py              # SID Stage A/B/C runtime 标记
│       ├── stage_* / run_stage_*         # 最新生成阶段因果 probes
│       └── results/                     # 指标、生成结果、profiling
├── paper_draft/                         # LaTex 论文草稿
├── PTQ_papers/                          # 本地量化论文，含 MLLM PTQ
├── data/                                # 数据不在 git 中
├── README.md                            # 同事快速使用 GPTQ 的中文说明
├── SID_Generation_Stage_Probe.md        # Stage A/B/C 初始设计
├── SID_Generation_Stage_Probe_Summary.md# 最新阶段/activation 总结
└── SESSION_HANDOFF.md                   # 本文件
```

## 3. 数据、模型和标准小实验协议

模型路径：

```text
/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B
```

数据根目录：

```text
data/onerec_data/benchmark-data-calib1024
```

用户已将原约 27k 测试数据拆成约 26k test 与 1024 条独立 calibration。校准集不得与 test 混用。

快速迭代的标准协议：

```text
task=ad
calib_split=calib
calib_sample_size=128
sample_size=1000
num_beams=32
num_return_sequences=32
max_new_tokens=3
```

主指标通常看 SID `Pass@K`、`Recall@K`，重点为 `K=32`。对 test1000，除了 aggregate metric，还应报告同一 W8A8 baseline 下的 paired recovery：

```text
recovery   = baseline failed, variant succeeded
regression = baseline succeeded, variant failed
net gain   = recovery - regression
```

环境与运行注意事项：

- Python：`/home/yhhuang/miniconda3/envs/benchmark2/bin/python`，PyTorch 为 `2.9.0+cu128`。
- 先确认 `python -c 'import torch'` 成功；单独 `conda activate benchmark2` 可能因 shell 未 init 而失败。
- real FP8 依赖 GPU、`torch._scaled_mm` 和 vLLM fused `scaled_fp8_quant`。
- 运行前使用 `nvidia-smi` 检查空卡。GPTQ/calibration 是离线过程，正式时延不能计入它们的时间。

## 4. 基础量化与 real runtime

`real_quant/naive_w8a8/modules.py::RealFP8Linear` 是 real quant 核心：

- Weight：离线量化，FP8 per-output-channel scale。
- Activation：默认 dynamic per-token FP8 quantization。
- GEMM：FP8 `torch._scaled_mm`。
- QKV 和 gate/up：`apply.py` 重写 attention/MLP forward，让共享输入量化一次。
- Decode A16：`seq_len=1` decode 跳过 activation FP8 quantization，权重仍为 FP8-QDQ。

已确认的工程结论：

1. 初始 naive W8A8 端到端加速不理想，主要不是 FP8 GEMM 无效，而是 dynamic activation quantization 的 absmax/scale/quantize 和额外 kernel launch 开销。
2. 已接入 vLLM fused dynamic quantization `scaled_fp8_quant`，消除了原先显著的 `abs/div/clamp/to(float8)` 碎 kernel 开销。
3. `torch._scaled_mm` 在大 prefill matrix 上相对 BF16 Linear 有明显优势；短 decode matrix 太小，吃不到 FP8 GEMM 红利，dynamic quantization 反而可能变慢。
4. 推荐只生成 3 个 SID token，decode 仅两次。因此 W8A8 prefill + W8A16 decode 是合理工程策略：时延代价小，且减少 decode activation quantization 的数值损失。

Decode A16 是工程配置，不应轻易包装成论文算法创新；但算法对比时必须固定是否启用它。

## 5. GPTQ、GPTAQ 与术语

### Plain GPTQ

每层 Linear 用输入二阶统计 `H = X^T X`，再使用标准 GPTQ 的 Cholesky/blockwise 误差补偿量化每个 weight row。当前 real runtime 已基本对齐传统 GPTQ 的 row 顺序、Cholesky 和 block 补偿。

### GPTAQ

仓库实现了 GPTAQ 风格非对称补偿/更新，核心在 `fake_quant_learnable/gptq.py` 与 `real_quant/naive_w8a8/gptq_runtime.py`。它支持 `alpha`，当前默认应为 `1.0`。GPTAQ 能在公式上更自然地处理 activation-aware 的量化目标，但它优化的是 prefill W8A8；decode A16 会改变实际推理路径，不能假设 GPTAQ 必然优于 GPTQ + decode A16。

### 命名约定

历史对话中有混淆，后续必须统一：

- `gptq` / `plain GPTQ`：普通 `H=X^T X`。
- `weighted GPTQ` 或 `gptq weight`：token-group weighted Hessian 的 GPTQ 变体；历史上包括 role/slot group 与 CE/多 target 梯度聚合权重。
- `GPTAQ`：GPTAQ 权重量化/补偿框架，不等同于 weighted GPTQ。
- `conditional GPTQ`：token group x output-channel 的 hard assignment，让不同 weight row 使用不同 group Hessian。

每次实验必须明确：token group 类型、梯度 loss 是 `sid_a CE` 还是 full-SID multi-target、是否 activation-aware、是否 decode A16、是否 tail1。

## 6. 已探索的 token 异质性方向

### 表示与 activation probes

已完成的观察：

- 输入 embedding t-SNE：text token 与 SID token 有明显 gap；SID-a/b/c 整体更紧聚，高层中同一 slot 还可分出 history 与 interest 小簇。
- layer-wise 表示可视化：slot 分组比单纯 history/interest 更稳定，也更适合作为 SID 结构叙事。
- token-wise activation outlier channel：不同 group 的 top-k outlier channel 不完全一致。正确做法是 token-wise top-k 后再聚合，不能先 group 平均后取 top-k。
- channel-energy、Hessian、channel sensitivity、conditional Hessian probes 均说明 token group x channel 存在差异，但“有差异”不自动推出某个 Hessian 操作会提升推荐指标。

相关文件：

```text
fake_quant_learnable/probe_slot_activation_channel_gap.py
fake_quant_learnable/probe_slot_outlier_channels.py
fake_quant_learnable/results/analysis/slot_activation_channel_gap/
fake_quant_learnable/results/analysis/slot_tokenwise_outlier_channels/
```

### Token-weighted GPTQ

曾按 CE/end-loss 梯度对 token group 聚合，再以对角 token 权重修改 Hessian；也试过手工 role/slot 权重。

结论：早期全量实验中，手工 role 权重和部分 group-gradient weighted GPTQ 曾接近或略优于 plain GPTQ，slot 手工权重在 K=32 也有收益。但结果不稳定：换成 GPTAQ、activation-aware 目标或小数据协议后，plain GPTAQ 有时优于 slot-weighted GPTAQ。直接由 end loss/CE 梯度得到权重的逻辑不够紧，因为信号距离每层重建目标较远；逐 token 细粒度权重也曾劣于 group 聚合。

因此 weighted GPTQ 目前只能作为 baseline/ablation，不能视为已确认的论文主方法。

### Conditional GPTQ

动机是不同 group 对不同 output channel 的响应可能不同。它按 token group x channel hard assignment，让不同 weight row 使用不同 group Hessian，实现在 `real_quant/naive_w8a8/conditional_gptq.py`。

结果：部分小/全量比较中 conditional GPTQ 比 plain GPTQ 略好，但 decode A16 收益不稳定，整体增益不足以成为可靠主线。方法说明在 `Conditional_Hessian_GPTQ_Method.md`。

## 7. 最新、最可信结果：SID generation stage probe

完整记录在 `SID_Generation_Stage_Probe_Summary.md`。新会话应优先阅读它。

### 阶段定义

| 阶段 | 模型处理位置 | 预测目标 |
| --- | --- | --- |
| Stage A | prefill 最后 `<|sid_begin|>` | SID-a |
| Stage B | decode 第一个生成 SID-a | SID-b |
| Stage C | decode 第二个生成 SID-b | SID-c |

这是生成因果位置，不是 prompt 中所有 sid_a/b/c token 的静态分组。

### 阶段级 activation rescue

Plain GPTQ FP8 权重保持固定，仅把选中阶段 activation 恢复为 BF16：

| Variant | Pass@32 | Recall@32 | paired net gain |
| --- | ---: | ---: | ---: |
| W8A8 | 0.172 | 0.058266 | - |
| Rescue-A | 0.184 | 0.061013 | +12 |
| Rescue-B | 0.177 | 0.061379 | +5 |
| Rescue-C | 0.179 | 0.059652 | +7 |
| Rescue-All | 0.182 | 0.064141 | +10 |

**Stage A 是最强的单阶段量化瓶颈。** Rescue-All Recall 更高但 Pass@32 低于 Rescue-A，说明 beam trajectory 下不同阶段收益不具可加性。

### Stage-A 权重/激活归因

real runtime 保留原始 BF16 权重快照：

| Mode | Stage-A weight | Stage-A activation | Pass@32 | Recall@32 | net gain |
| --- | --- | --- | ---: | ---: | ---: |
| W8A8 | GPTQ FP8-QDQ | FP8-QDQ | 0.172 | 0.058266 | - |
| A16 | GPTQ FP8-QDQ | BF16 | 0.184 | 0.061013 | +12 |
| W16 | BF16 | FP8-QDQ | 0.174 | 0.057510 | +2 |
| WA16 | BF16 | BF16 | 0.181 | 0.064276 | +9 |

结论：Stage-A 的主要可验证误差源是 **activation QDQ**，不是 weight QDQ。不要仅因为 Stage A 重要就直接设计 Stage-A GPTQ Hessian。

### Stage-A channel 误差与因果对照

真实 W8A8 runtime 中，Stage-A top 1% input channel 平均承载 36.0% activation QDQ 误差，top 5% 承载 60.3%。其误差与既有 BF16 token-wise profile 的描述性对齐更接近 boundary group。

但这不能直接形成方法。因果对照如下：

| Variant | Pass@32 | Recall@32 | paired net gain |
| --- | ---: | ---: | ---: |
| W8A8 | 0.172 | 0.058266 | - |
| 恢复误差 top-1% channel | 0.175 | 0.061055 | +3 |
| 恢复随机 top-1% channel | 0.177 | 0.063256 | +5 |

单个随机 seed 不足以证明随机选择更优，但足以否定“局部 QDQ error top-k 是可信的任务关键 channel 规则”。局部 MSE 不能刻画扰动经后续网络、SID-a candidate ranking、beam pruning 后的推荐指标影响。

## 8. 不应作为主线的结论

1. 不要把 tail1 作为论文核心方法。它在每层保护 prefill 最后 token，并不等价于保护最终 logits；下一层已混合 attention。它可作历史上界/ablation，但会损失时延且叙事不干净。
2. 不要直接用 activation QDQ error 最大的 channel 做保护/平滑；最新随机对照不支持。
3. 不要将 group Hessian 简单平均/加权后就声称保护 token 异质性；那只是新整体重建目标，未证明与推荐指标对齐。
4. 不要用 fake Stage-A attribution 得出因果结论。fake W8A8 baseline 与 real baseline 不一致，且 fake A16 与 real A16 符号相反。
5. 不要假设 GPTAQ 一定超过 GPTQ，或 activation-aware GPTAQ 一定与 decode A16 兼容。

## 9. 当前边界与建议下一步

当前最可靠的动机链：

```text
SID recommendation has structured token heterogeneity
  -> probes show representation/channel differences
  -> causal generation probe localizes strongest sensitivity to Stage A
  -> sensitivity is primarily activation-side
  -> naive local-QDQ-MSE channel protection fails against random control
```

该链是可靠分析结论，但**尚不足以构成新的量化算法**。

若继续探索 activation-side 方法，下一步必须先验证任务相关 selection criterion：基于 Stage-A 的 SID-a logit margin、candidate ranking 或 beam survival 对 channel 扰动的敏感度，而不是输入 QDQ MSE。先在小样本协议、多个随机对照下验证所选 channel 是否稳定优于随机选择；不能通过就不应继续堆叠 channel protection。

另一条保守路线是将工作收敛为 SID recommendation PTQ analysis + 工程部署研究，以 plain GPTQ/优选 weighted GPTQ 作 baseline，而不在无稳定增益时声称 token-aware Hessian 是主创新。

## 10. 论文草稿状态

论文草稿在 `paper_draft/`：

- `2_related_work.tex`：已写 LLM-based recommendation 与 PTQ 英文内容。叙事为：text item representation 缺失协同信号，SID tokenization 逐渐成为主流；OneRec 是端到端可部署的 SID generative recommender。
- `3_preliminaries.tex`：包含 LLM-based recommendation with Semantic IDs 与 GPTQ-based PTQ 初稿。无需强调 zero point，当前方法不涉及它。
- `1_introduction.tex`：存在用户写的开头/中文注释，应保留其叙事结构。不要上来强调 decode A16；它更适合实验/工程部分。

论文中避免过度宣称：当前最强实证是 Stage-A activation sensitivity，并非已经提出有效 token-aware PTQ 方法。

## 11. 常用入口与结果位置

普通 real GPTQ flags 可能演化，先查看：

```bash
python -m real_quant.naive_w8a8.run_hf_naive_w8a8 --help
```

最新 runner：

```text
real_quant/naive_w8a8/run_sid_stage_probe_fixed.py
real_quant/naive_w8a8/run_stage_a_weight_attribution.py
real_quant/naive_w8a8/probe_stage_a_activation_error.py
real_quant/naive_w8a8/run_stage_a_channel_rescue.py
```

最新结果：

```text
real_quant/naive_w8a8/results/probes/sid_generation_stage_ad_1p7b_gptq_calib128_test1000/
real_quant/naive_w8a8/results/probes/stage_a_weight_attribution_ad_1p7b_gptq_calib128_test1000/
real_quant/naive_w8a8/results/probes/stage_a_activation_error_ad_1p7b_gptq_calib128/
real_quant/naive_w8a8/results/probes/stage_a_channel_rescue_ad_1p7b_gptq_calib128_test1000/
```

复现实验前，先核对 W8A8 baseline 是否复现小协议：`Pass@32=0.172`、`Recall@32=0.058266`。不一致时先排查模型、split、calib、beam 和 runtime 路径，不要直接解释为算法差异。

## 12. 代码修改注意事项

- 工作树可能有用户未提交修改，禁止 reset/checkout/revert 无关文件。
- QKV/gate-up shared path 是易错点。任何 activation hook/patch 必须覆盖 `apply.py::_shared_prepare_input` 触发的路径。
- 新 probe 应独立放置，不污染普通 `run_hf_naive_w8a8.py` 默认行为。
- stage rescue 的首次实现仅 patch `RealFP8Linear.forward`，实际 W8A16 调用数为 0；修复后才覆盖共享路径。以后 runtime patch 必须 smoke test 调用次数或数值差异。
- dynamic activation quantization 的 runtime 与 fake quant 不完全等价。算法筛选若依赖 fake，必须先验证与 real path 的精度排序一致。
