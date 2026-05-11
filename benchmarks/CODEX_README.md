# OpenOneRec FP8 QDQ experiment notes

## Experiment goal
本阶段目标是验证 OneRec-1.7B 的部分权重被压到 FP8 e4m3 有效精度后，OpenOneRec benchmark 指标是否明显下降。

当前实验是 weight QDQ，不是真正的低精度推理：

```text
BF16 weight -> FP8 e4m3 quantize -> dequantize back to BF16 -> save normal HF checkpoint -> vLLM BF16 path
```

因此本阶段只回答“数值精度下降是否影响推荐指标”，不回答真实 FP8 kernel 的显存、吞吐或时延收益。

## Recommendation prediction scenario
当前重点不是开放式文本生成、物品理解或推荐理由生成，而是 prediction 类推荐任务：

```text
输入：用户历史行为序列 prompt
输出：一个 item 的 SID
目标：通过 beam search 得到 top-k SID 候选，用于 pass@k / recall@k 等推荐指标
```

推荐任务会在 prompt 末尾追加：

```text
<|sid_begin|>
```

然后模型只继续生成 SID code token。当前 AD full 实验中 benchmark 命令覆盖为：

```text
num_beams = 32
num_return_sequences = 32
max_new_tokens = 3
enable_thinking = false
```

因此实际生成不是自然语言回答，而是：

```text
history prompt + <|sid_begin|> -> 3-token SID sequence
```

vLLM 路径中，`num_beams` 会触发 `BeamSearchParams`，然后调用 `llm.beam_search(...)`。每个样本返回多个 beam candidate 及其 `cum_logprob`。当前 recommendation evaluator 默认：

```text
select_k = first_k
```

也就是直接使用 beam search 返回的候选顺序作为 top-k 推荐顺序；当前 AD 评测主要看 top-1 和 top-32。

这意味着后续量化分析应重点关注：

```text
SID beam candidate set 是否稳定
top-k SID 顺序是否稳定
target SID rank 是否变化
target SID 是否从 top-32 掉出
3 个 SID code token 哪个位置更容易受量化影响
```

不应把物品理解、自然语言生成质量或通用 LM loss 当作当前阶段的主要优化目标。

benchmark 的最终汇总文件 `eval_results.json` 只包含聚合指标，不足以分析 beam ranking。逐样本分析应读取：

```text
<result_dir>/<model_name>/ad/test_generated.json
```

其中每个 sample 主要包含：

```text
generations    : 32 个 beam 返回的 SID candidate
logprobs       : 与 generations 一一对应的 cumulative logprob，数值越大越好
ground_truth   : 一个或多个正确 SID，格式为 <|sid_begin|>...<|sid_end|>
metadata       : 包含 answer_pid 等 PID ground truth
pid_generations: generations 映射到 PID 后的推荐列表
pass@k / recall@k / pid_*: evaluator 写回的逐样本指标
```

`ground_truth` 有多个 SID 是正常的，evaluator 会解析成正确 SID 集合；`pass@k` 只要 top-k 命中任意正确 SID 即为 true，`position1_pass@k` 只看第一个正确 SID，`recall@k` 是 top-k 命中的正确 SID 数量除以正确 SID 集合大小。整体指标是所有样本的平均或 true 比例。

PID 指标与 SID 指标口径相同，但会先通过 `sid2pid.json` 把生成 SID 映射成 PID，再与 `metadata.answer_pid` 比较：

```text
pid_pass@k
pid_position1_pass@k
pid_recall@k
```

因此后续 beam/rank 分析应以 `test_generated.json` 为主，比较 baseline 和 QDQ 模型的：

```text
top-1 是否变化
top-32 SID/PID set overlap
target SID/PID rank shift
target 是否从 top-32 掉出
top1-top2 logprob margin 是否变小
```

## Model and benchmark
原始模型：

```bash
/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B
```

模型结构摘要：

```text
architecture: Qwen3ForCausalLM
dtype: bfloat16
layers: 28
hidden_size: 2048
intermediate_size: 6144
attention heads: 16
kv heads: 8
vocab_size: 176384
total parameters in checkpoint: 2,131,878,912
```

主要参数构成：

```text
embed_tokens:      361,234,432
lm_head:           361,234,432
MLP gate_proj:     352,321,536
MLP up_proj:       352,321,536
MLP down_proj:     352,321,536
attention q_proj:  117,440,512
attention o_proj:  117,440,512
attention k_proj:   58,720,256
attention v_proj:   58,720,256
```

当前已完成的评测是 `ad` full set：

```text
total_samples: 27,677
baseline result: results/v1.0/OneRec-1.7B_ad_full/eval_results.json
summary table: results/v1.0/fp8_qdq_ad_full_summary.md
summary csv: results/v1.0/fp8_qdq_ad_full_summary.csv
```

用于 beam/rank 快速分析的 AD 1000 条子集：

```text
data/onerec_data/benchmark-data/ad/ad_test_sample_1000.parquet
```

## Experiment setup
QDQ 脚本：

```text
scripts/quantize_qdq_fp8.py
```

批量实验脚本：

```text
scripts/run_fp8_qdq_experiments.sh
```

结果汇总脚本：

```text
scripts/summarize_fp8_qdq_results.py
```

beam/rank 对比脚本：

```text
scripts/analyze_beam_rank_shift.py
```

QDQ 默认设置：

```text
fp8_format: e4m3
scale_granularity: per_row
output_dtype: same
device: cpu
```

`per_row` 表示每个 Linear weight 的每个输出通道单独计算 scale，再做 FP8 e4m3 QDQ。输出 checkpoint 仍然是普通 BF16 HuggingFace checkpoint，可以直接给当前 vLLM benchmark 使用。

运行一组实验示例：

```bash
python scripts/quantize_qdq_fp8.py \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --output_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B-fp8e4m3-gate-up \
  --target_regex 'model\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight$' \
  --fp8_format e4m3 \
  --scale_granularity per_row
```

批量生成 checkpoint 并串行跑 benchmark：

```bash
RUN_EVAL=1 bash scripts/run_fp8_qdq_experiments.sh
```

跑 AD 1000 条子集 benchmark：

```bash
RUN_EVAL=1 SAMPLE_SIZE=1000 EXPERIMENTS="gate_only up_only" bash scripts/run_fp8_qdq_experiments.sh
```

设置 `SAMPLE_SIZE=1000` 后，`eval_script.sh` 会向 evaluate 传入：

```text
--sample_size 1000
```

loader 会优先读取：

```text
data/onerec_data/benchmark-data/ad/ad_test_sample_1000.parquet
```

默认结果后缀会从 `ad_full` 自动变成：

```text
ad_sample_1000
```

例如：

```text
results/v1.0/results_OneRec-1.7B-fp8e4m3-gate-only_ad_sample_1000/
```

汇总结果：

```bash
python scripts/summarize_fp8_qdq_results.py
```

对比 AD 1000 子集的 baseline、`mlp-all` 和 `block-linears` beam/rank：

```bash
python scripts/analyze_beam_rank_shift.py
```

默认输出：

```text
results/v1.0/beam_rank_ad_sample_1000_samples.csv
results/v1.0/beam_rank_ad_sample_1000_summary.csv
results/v1.0/beam_rank_ad_sample_1000_summary.md
```

当前 `baseline`、`mlp-all`、`block-linears` 的 AD 1000 子集 beam/rank 结论整理在：

```text
results/v1.0/beam_rank_ad_sample_1000_analysis.md
```

核心结论是：FP8 e4m3 weight QDQ 主要造成低幅度 beam 边界扰动，top-1 变化较多，但 top-32 candidate set 仍保留约 24-25/32 个 baseline 候选；`lost` 和 `gained` 都主要发生在 top-32 后半段。

也可以显式增加其他模型：

```bash
python scripts/analyze_beam_rank_shift.py \
  --compare gate-only=results/v1.0/results_OneRec-1.7B-fp8e4m3-gate-only_ad_sample_1000
```

## Quantization settings
本阶段测试了以下 FP8 e4m3 QDQ 设置，暂未量化 `embed_tokens` 和 `lm_head`。

```text
gate-only    : MLP gate_proj
up-only      : MLP up_proj
gate-up      : MLP gate_proj + up_proj
mlp-down     : MLP down_proj
mlp-all      : MLP gate_proj + up_proj + down_proj
attn-o       : attention o_proj
attn-qvo     : attention q_proj + v_proj + o_proj
attn-all     : attention q_proj + k_proj + v_proj + o_proj
block-linears: all MLP gate/up/down + attention q/k/v/o
```

## Metric aggregation
`avg_accuracy_drop_pct` 是相对 baseline 的平均精度下降百分比，使用下面 12 个 ad 指标计算：

```text
pass@1
position1_pass@1
recall@1
pass@32
position1_pass@32
recall@32
pid_pass@1
pid_position1_pass@1
pid_recall@1
pid_pass@32
pid_position1_pass@32
pid_recall@32
```

单个指标下降比例：

```text
drop_pct = (baseline_metric - quant_metric) / baseline_metric * 100
```

`avg_accuracy_drop_pct` 是上述 12 个 `drop_pct` 的简单平均。负数表示该量化配置在本次 ad full 单次评测中平均略高于 baseline，建议先按评测波动理解。

`quantized_param_ratio_pct` 表示经过 QDQ 的参数占全模型 checkpoint 参数比例。由于输出仍是 BF16 checkpoint，这不是实际磁盘或显存压缩比例。

## Current results
阶段性结果如下，完整表见：

```text
results/v1.0/fp8_qdq_ad_full_summary.md
results/v1.0/fp8_qdq_ad_full_summary.csv
```

| quant_setting | quantized_param_ratio_pct | avg_accuracy_drop_pct | comment |
| --- | ---: | ---: | --- |
| baseline | 0.00 | 0.0000 | 原始 BF16 模型 |
| gate-only | 16.53 | 2.8934 | 单独量化 gate_proj 已有明显下降，说明 gate 侧较敏感 |
| up-only | 16.53 | 2.0292 | 单独量化 up_proj 也有下降，但弱于 gate-only |
| gate-up | 33.05 | 3.0571 | 初始实验；下降比后续组合更明显 |
| gate-up-rerun1 | 33.05 | 3.0571 | 从量化开始重跑；12 个 accuracy 指标与 gate-up 完全一致 |
| mlp-down | 16.53 | 0.0839 | 几乎无平均下降 |
| mlp-all | 49.58 | 1.0445 | 覆盖近半参数，平均下降约 1% |
| attn-o | 5.51 | 0.7019 | 小范围 attention 输出投影量化，下降较小 |
| attn-qvo | 13.77 | -1.6550 | 本次单次评测平均略高于 baseline，应按波动看 |
| attn-all | 16.53 | 1.3679 | 完整 attention projection 量化，下降约 1.37% |
| block-linears | 66.11 | 1.9479 | 覆盖 2/3 参数，平均下降约 1.95% |

初步观察：

1. FP8 e4m3 per-row QDQ 对 OneRec-1.7B 的 AD benchmark 比较温和。
2. `mlp-all` 的性价比较好：量化约 49.58% 参数，平均下降约 1.04%。
3. `block-linears` 覆盖约 66.11% 参数，平均下降仍低于 2%，值得作为下一阶段重点候选。
4. `gate-up-rerun1` 从量化开始重跑后，12 个 accuracy 指标与原 `gate-up` 完全一致，说明 `gate-up` 的 3.06% 平均下降不是单次 benchmark 随机波动造成的。
5. `gate-only` 平均下降 2.89%，`up-only` 平均下降 2.03%，说明 `gate-up` 的异常主要来自 gate/up 两个输入侧 MLP 投影，其中 gate 侧更敏感。
6. `gate-up` 的下降 3.06% 并不是 `gate-only + up-only` 的简单相加；而 `mlp-all` 反而只下降 1.04%，说明加入 `down_proj` QDQ 后可能改变了扰动方向，对 beam ranking 有一定抵消效果。
7. 当前 QDQ 实验没有真实性能收益；如果指标稳定，下一步才考虑 vLLM 原生 FP8/低精度部署路径。

## Real FP8 inference results
除了 QDQ simulation，也补充了真实 FP8 推理实验：

```text
vLLM online FP8         : LLM(..., quantization="fp8")
llm-compressor FP8_DYNAMIC: 离线生成 vLLM-compatible FP8 checkpoint
llm-compressor FP8 static activation: calibration samples 确定 activation scale
```

当前理解：

```text
vLLM online FP8: Linear weight 使用 FP8 per-tensor scale，activation 动态量化。
llm-compressor FP8_DYNAMIC: Linear weight 静态 per-channel，activation 动态 per-token。
llm-compressor FP8_ACT_TENSOR_STATIC: Linear weight 静态 per-channel，activation 静态 per-tensor。
```

AD full 结果（修正 vLLM 环境后的双卡 H100 / L20X）：

| setting | result_dir | avg_accuracy_drop_pct | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | total_time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline BF16 | `results/v1.0/results_OneRec-1.7B-baseline` | 0.00 | 0.21595549 | 0.07480481 | 0.20218954 | 0.06944162 | 28.26 min |
| FP8_DYNAMIC, act dynamic per-token | `results/v1.0/results_OneRec-1.7B-llmcompressor-fp8-dynamic` | 2.80 | 0.20627236 | 0.07106940 | 0.19369874 | 0.06606428 | 27.96 min |
| FP8_ACT_TENSOR_STATIC, act static per-tensor | `results/v1.0/results_OneRec-1.7B-llmcompressor-fp8-act-tensor-static` | 5.06 | 0.20424902 | 0.07010798 | 0.19080825 | 0.06486172 | 27.79 min |

注：`avg_accuracy_drop_pct` 按 `pass@1 / recall@1 / pass@32 / recall@32 / pid_pass@32 / pid_recall@32` 相对 baseline 的平均下降计算。

AD sample 1000 profiling（双卡，`PROFILE_TIMING=true`）：

| item | baseline | FP8_DYNAMIC | time reduction | speedup |
| --- | ---: | ---: | ---: | ---: |
| `worker_wall_time_max` | 68.92s | 65.79s | 4.53% | 4.74% |
| `batch_total_time` worker sum | 136.95s | 129.92s | 5.13% | 5.41% |
| `vllm_generate_time` worker sum | 133.85s | 127.25s | 4.93% | 5.19% |
| `tokenize_time` | 1.23s | 1.19s | 3.50% | 3.62% |
| `decode_time` | 0.18s | 0.18s | 1.14% | 1.15% |
| `pack_time` | 0.0101s | 0.0104s | -3.06% | -2.97% |

Profiling 占比：

| component | baseline | FP8_DYNAMIC |
| --- | ---: | ---: |
| tokenize | 0.91% | 0.92% |
| vLLM beam search / generate | 98.94% | 98.93% |
| decode | 0.14% | 0.14% |
| pack | 0.01% | 0.01% |

结论：当前时间开销几乎全部集中在 `vllm_generate_time`，也就是 vLLM 的 beam search / generate 黑盒中。FP8_DYNAMIC 在该黑盒上约有 5% speedup，但收益较小，full benchmark 中容易被调度和运行波动抵消。后续 latency 分析应继续拆 vLLM 内部的 prefill、3-step beam expansion、top-k/logprob 维护和 kernel path 开销。

Additional implementation notes:

1. `fp8_weight_only_channel` 是可由 llm-compressor / compressed-tensors 表达的自定义配置：

```text
weight: FP8 e4m3, per-channel, static
activation: BF16 / FP16, not quantized
```

但当前 vLLM 环境无法稳定加载该 checkpoint。vLLM 会将它识别为 `compressed_tensors_w8a16_fp8`，并进入 FP8 Marlin weight-only 路径；H100 / L20X 上出现如下 warning 和 fatal error：

```text
Your GPU does not have native support for FP8 computation but FP8 quantization is being used.
Weight-only FP8 compression will be used leveraging the Marlin kernel.

torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

这个 warning 已有类似 GitHub issue 讨论：`https://github.com/vllm-project/vllm/issues/10142`。这里不应理解为 H100 不支持 FP8，而是 vLLM 对 weight-only FP8 config 走了 Marlin backend。`VLLM_USE_V1=0` 后仍失败，因此当前不把真实 vLLM weight-only FP8 作为可跑实验路径；FP8 weight-only 的精度消融仍以 QDQ 为主。

2. 下一组真实 FP8 实验优先比较 activation quantization path：

```text
FP8_DYNAMIC:
  weight FP8 per-channel static
  activation FP8 dynamic per-token

fp8_weight_channel_act_tensor_static:
  weight FP8 per-channel static
  activation FP8 static per-tensor, calibrated on AD prompts
```

校准数据已导出到：

```text
quantization_configs/calibration/ad_sample_1000.jsonl
```

这组实验用于比较在线 activation dynamic quantization 与离线 calibrated static activation quantization 的精度和时延差异。

阶段性结论：

```text
两种真实 llm-compressor FP8 的耗时都约 28 min，和 BF16 baseline 基本持平，没有稳定 latency 收益。
FP8_DYNAMIC 的精度优于静态 activation per-tensor：平均下降 2.80% vs 5.06%。
离线 calibrated static per-tensor activation scale 并没有改善推荐指标，反而比 dynamic per-token 更伤 top-32 / PID recall。
当前更像是 activation scale 粒度问题，而不是简单“离线校准一定更稳”；per-token dynamic scale 对 SID beam ranking 更友好。
```

## Next questions
当前 QDQ 实验已经说明 OneRec-1.7B 对 FP8 e4m3 weight-only 量化比较耐受，而真实 W8A8 FP8 会带来更明显的推荐指标下降。下一阶段更应该围绕“SID prediction / top-k recommendation 的量化敏感性”展开，而不是简单套用 AWQ/GPTQ。

可以把后续研究问题收敛为：

```text
通用 LLM 量化方法是否适合推荐大语言模型？
推荐 SID token、3-token SID 生成和 beam-search top-k 排序会不会带来不同的量化敏感性？
```

### Recommended next step: activation profiling

当前已完成 weight-only QDQ 和 beam/rank 分析。阶段性结论是：FP8 e4m3 weight QDQ 主要造成 top-32 边界扰动，而不是整体推荐能力崩坏。

下一步不应继续大量枚举 weight-only 量化配置，而是先探索 activation quantization 是否值得做。核心问题是：

```text
推荐边界样本、SID token 位置和 SID decode 阶段的 activation 分布是否有特殊性？
如果有，activation quantization 可能需要推荐场景感知的 scale / clipping / mixed precision 策略。
如果没有，activation quantization 可能只能按通用 LLM 方案推进。
```

建议先只 profile baseline 模型，避免一开始把量化噪声混进去。样本从 AD 1000 beam/rank 结果中分组抽取：

```text
stable_hit_front      : baseline target rank 1-5，量化后仍命中
boundary_lost         : baseline target rank 20-32，量化后掉出 top-32
boundary_gained       : baseline 未命中，量化后 target rank 17-32
both_miss             : baseline 和量化都未命中
baseline_good_multi_gt: baseline pass@1=True，且 ground_truth SID 数量 > 1
```

每组先取 20-50 条即可，不需要全量 profiling。

建议分 token/阶段统计：

```text
history prompt tokens
历史 SID tokens
<|sid_begin|> 位置
第 1 个生成 SID code token
第 2 个生成 SID code token
第 3 个生成 SID code token
```

建议先看的模块：

```text
每层 residual hidden state
MLP input
MLP gate_proj output
MLP up_proj output
MLP down_proj output
attention o_proj output
```

建议统计量：

```text
mean / std
absmax
p99 / p99.9
outlier ratio: abs(x) > threshold
per-channel absmax
```

第一阶段目标不是直接提出 activation quantization 方法，而是判断是否存在值得利用的推荐相关现象，例如：

```text
boundary_lost 样本是否在特定层/模块有更高 activation outlier
SID decode 第 1/2/3 步是否比 history prompt 更敏感
历史 SID token 与普通文本 token 的 activation 分布是否不同
gate/up/down activation 是否和 target 掉出 top-32 更相关
```

若发现明显差异，下一步再考虑 activation fake quant、SmoothQuant-style scaling、SID-decode-aware scaling 或 mixed precision；若差异不明显，则 activation quantization 可能不应作为主要创新方向。


# 当前问题
1. 量化模型时延收益微弱（5%左右），需要分析各部分执行时间查问题，如果用上FP8乘法做前向计算应该不至于收益这么低
2. 有尝试weight-only fp8量化（W8A16），但是这块vllm推理会报错，网上也有人反馈这个问题，怀疑是8位和16位数据计算有bug，暂时无法避免，跑W8A8的量化就没有这个问题
3. 逐模块最简量化方案太简单了，如果是