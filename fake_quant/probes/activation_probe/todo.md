# Activation Probe TODO

目标：围绕推荐 LLM 的两个特有假设继续分析激活值分布。

1. OneRec 经过增训学到了 SID token，SID token 可能具有不同于普通文本 token 的激活分布。
2. AD SID prediction 的输出长度固定为 `a -> b -> c -> end`，如果固定 decode position 的激活分布稳定，后续可以设计 position-aware 的离线 activation quantization 参数。

## Existing Basis

当前已有脚本：

```text
profile_ad_activations.py
  - 支持 token group 统计：prompt_text / chat_special / sid_boundary / sid_code / generated_sid_code 等。
  - 支持 prefill 和 greedy decode_step 的激活统计。
  - 输出 event_stats.csv、summary_by_stage_token_group_module.csv、top_outliers.csv。
  - 可选输出 log-binned histogram plots。

plot_token_channel_activations.py
  - 单样本 token-channel heatmap / 3D surface。
  - 默认节点：attn_qkv_input / attn_o_input / ffn_gate_up_input / ffn_down_input。

analyze_channel_overlap.py
  - 单样本、跨层 top outlier channel overlap。
  - 支持 mean_abs / p99_abs / max_abs。
```

已有结果说明：

- token group 维度已经初步看到 chat special token 有异常高单值。
- 单样本 3D 图显示异常值更多体现为 channel-wise outlier，而不是 token-wise outlier。
- 单样本 channel overlap 显示 attn/ffn 输入侧的异常 channel 有明显重叠。

当前缺口：

- decode profiling 目前是 greedy 生成，不是 teacher-forcing，因此 `decode_step_1/2/3` 不一定严格对应 `a/b/c/end` 的 ground-truth 位置。
- channel overlap 主要是单样本分析，还没有系统验证跨样本稳定性。
- 还没有直接验证 `global static scale`、`decode-step-wise static scale` 和 `dynamic per-token scale` 的量化误差差异。

## TODO 1: Re-run Token Group Distribution With Larger Samples

目的：先复用已有 `profile_ad_activations.py`，确认 SID token 与普通 token 的总体分布差异是否稳定。

建议配置：

```bash
SAMPLE_SIZE=128 SAVE_HISTOGRAMS=true \
bash fake_quant/probes/activation_probe/run_activation_profile.sh
```

重点看：

- `summary_by_stage_token_group_module.csv`
- `activation_histograms.csv`
- histogram plots

对比 token group：

```text
prompt_text
chat_special
sid_boundary
sid_code
sid_boundary_next_token
generated_sid_code
```

重点节点：

```text
attn_input_norm
mlp_input_norm
self_attn.q_proj / k_proj / v_proj / o_proj
mlp.gate_proj / up_proj / down_proj
residual_block_output
final_norm
```

判断标准：

- SID token 的 `absmax / p99 / p999 / mean_abs` 是否系统性不同于普通文本 token。
- SID token 是否在高层出现更强 outlier。
- chat special token 的异常单值是否只集中在少数层/节点，避免后续误把 chat special 的 outlier 当成 SID 规律。

## TODO 2: Add Teacher-Forcing Decode-Step Profiler

目的：严格对齐固定生成位置，分别统计 `predict_a / predict_b / predict_c / predict_end` 的激活分布。

需要新增或扩展脚本：

```text
fake_quant/probes/activation_probe/profile_teacher_forced_decode_steps.py
```

数据构造：

```text
step predict_a:
  input = prompt + <|sid_begin|>
  target = gt_a

step predict_b:
  input = prompt + <|sid_begin|> + gt_a
  target = gt_b

step predict_c:
  input = prompt + <|sid_begin|> + gt_a + gt_b
  target = gt_c

step predict_end:
  input = prompt + <|sid_begin|> + gt_a + gt_b + gt_c
  target = <|sid_end|>
```

实现细节：

- 一条样本如果有多个 ground-truth SID，第一版先固定使用第一个 SID sequence。
- 使用 KV cache 做逐步 teacher-forcing，避免每个 step 重跑完整 prompt。
- 每个 step 只统计当前用于预测下一个 SID token 的 last-token activation。
- stage 命名固定为 `predict_a`、`predict_b`、`predict_c`、`predict_end`。

重点 hook 节点：

```text
attn_qkv_input
attn_o_input
ffn_gate_up_input
ffn_down_input
residual_block_output
final_norm
```

建议先看层：

```text
0, 7, 14, 21, 27
```

输出：

```text
event_stats.csv
summary_by_step_layer_node.csv
summary_by_step_node.csv
channel_scores.pt
sample_summary.json
```

每个 step/layer/node 统计：

```text
mean_abs
p99_abs
p999_abs
max_abs
std
outlier_ratio@6/10/20
per-channel mean_abs
per-channel p99_abs
per-channel max_abs
```

## TODO 3: Cross-Sample Stability For Fixed Decode Positions

目的：验证固定位置的 activation outlier channel 是否跨样本稳定。

状态：已完成 `sample_size=128` 初步实验。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=4 SAMPLE_SIZE=128 \
bash fake_quant/probes/activation_probe/run_teacher_forced_channel_stability.sh
```

结果目录：

```text
fake_quant/probes/activation_probe/activation_profiles/v1.0/OneRec-1.7B-ad-teacher-forced-channel-stability-sample-128/
```

输入：

```text
TODO 2 生成的 channel_scores.pt
```

统计粒度：

```text
step x layer x node
```

指标：

```text
top-k Jaccard: top 1% / 5% / 32 channels
per-channel rank correlation
scale coefficient of variation: std(scale_c) / mean(scale_c)
calib-test top-k overlap
```

对比对象：

```text
predict_a vs predict_b vs predict_c vs predict_end
同一 step 跨样本
不同 step 之间
低层 vs 高层
attn input vs ffn input
```

预期结论：

- 如果同一 step 跨样本 Jaccard 高，说明该 position 可以使用固定离线 scale。
- 如果不同 step 之间 Jaccard 低，说明 `a/b/c/end` 不应该共享同一套 activation scale。
- 如果高层稳定性明显更低，后续可以只在低/中层使用 step-wise static scale，高层保留 dynamic 或更保守策略。

当前结论：

- `predict_a` 的 channel 稳定性非常高，selected layer/node 平均 top-5% Jaccard 为 `0.789`。这大概率来自固定输入 `<|sid_begin|>` 的 control-token 特征，因此 `predict_a` 单独使用 static scale 是有动机的。
- `predict_b/c/end` 的跨样本稳定性明显更弱，平均 top-5% Jaccard 分别为 `0.189 / 0.145 / 0.187`。虽然高于随机 top-5% Jaccard（约 `0.026`），但不足以直接证明 per-position static scale 一定可靠。
- high-layer 的 `predict_b/c` 尤其不稳定。例如 `layer 27 / residual_block_output` 的 top-5% Jaccard 只有 `0.039 / 0.032`，说明高层 item-specific channel variation 很强。
- 不同 step 的均值通道集合有结构：`predict_a` 与其它 step 的 top-5% overlap 较低（约 `0.16-0.19`），而 `predict_b` 和 `predict_c` 更接近（约 `0.489`）。因此如果做 position-aware scale，可能不是四套完全独立 scale，而是 `predict_a` 单独、`predict_b/c` 作为 continuation group、`predict_end` 另行处理。
- 结论不是“step-wise static activation quantization 已经成立”，而是“固定位置确实有不同分布，但只有 `predict_a` 的 top channel 强稳定；`predict_b/c/end` 需要继续用 QDQ error 验证 static scale 是否优于 global static”。

## TODO 4: Position-Aware Static Activation Scale Error Study

目的：直接验证固定输出位置能否带来更小的 activation quantization error。

比较 scale 策略：

```text
global_static:
  所有 decode step 共享同一个 node/layer activation scale。

step_static:
  predict_a / predict_b / predict_c / predict_end 分别使用不同 scale。

layer_node_step_static:
  每个 layer/node/step 单独 scale。

dynamic_per_token:
  当前 activation dynamic per-token baseline。
```

先只做数值误差，不直接跑完整推荐指标。

统计：

```text
MSE(x, QDQ(x))
relative_error = ||x - QDQ(x)|| / ||x||
SQNR
clipping_ratio
Linear output error = ||Linear(x) - Linear(QDQ(x))|| / ||Linear(x)||
```

判断标准：

- `step_static` 是否明显优于 `global_static`。
- `step_static` 是否接近 `dynamic_per_token`。
- 哪些 layer/node 对 step-wise scale 最敏感。

如果成立，后续可以把它作为推荐 LLM 的量化动机：

```text
传统 LLM decode 位置语义不固定，因此很难提前为每个输出位置设计 activation scale；
推荐 LLM 的 SID 输出位置固定，因此可以做 decode-position-aware offline activation quantization。
```

## TODO 5: Connect Distribution Findings To Quantization Method

根据前面结果决定后续算法方向：

```text
Case A: SID token 分布显著特殊，但 decode step 不稳定
  -> 做 SID-token-aware calibration / scale protection。

Case B: decode step 稳定，不同 step 分布不同
  -> 做 step-wise static activation quantization。

Case C: 只有部分 layer/node 稳定
  -> 做 layer/node selective static quantization，其它位置保留 dynamic。

Case D: SID / step 分布都不稳定
  -> 不继续做离线 activation scale，转向量化误差重构或 layer-wise precision allocation。
```

## Suggested Execution Order

1. 复用 `profile_ad_activations.py` 跑 `sample_size=128`，汇总 token group 分布。
2. 实现 teacher-forcing decode-step profiler。
3. 跑 `sample_size=128` 的 step-wise channel stability。
4. 做 global static vs step static vs dynamic 的 QDQ error 对比。
5. 如果 step-wise static 明显有效，再接入 fake quant 评测跑 `sample_size=1000`。
