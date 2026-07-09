# Real W8A8 时延探索总结

本文档总结当前 real quant 工程探索的主线：初始 naive W8A8 比 BF16 更慢，随后通过 profiling 拆解瓶颈，确认 `_scaled_mm` 在大矩阵上有潜在收益，但原始 dynamic activation quant 开销过高；最终通过 fused dynamic quant 和 decode A16 绕过小矩阵路径，把 W8A8 从负收益推进到正收益。

## 1. 初始现象：naive W8A8 比 BF16 更慢

最初的 real quant 实现使用：

- 权重：离线 per-output-channel FP8 E4M3 量化，推理时保存为 `weight_fp8_t + weight_scale`。
- 激活：推理时 per-token dynamic absmax 量化。
- GEMM：用 `torch._scaled_mm(x_fp8, weight_fp8_t, scale_a, scale_b)` 替换 `nn.Linear`。
- q/k/v 和 gate/up 做 shared input activation quant，避免同一输入重复量化。

1000 条 `ad` 子集的初始结果如下：

| 方法 | generate 总时长 | tokens/s | 相对 BF16 |
| --- | ---: | ---: | ---: |
| BF16 | 541.635 s | 177.241 | 1.000x |
| naive W8A8 dynamic | 651.489 s | 147.355 | 0.831x |

这个结果说明：虽然 W8A8 理论上降低了矩阵乘法的数据量，但端到端推理反而慢了约 17%。因此问题不应只看 GEMM 理论峰值，而要拆解整个 Linear 替换路径。

## 2. 第一轮定位：`_scaled_mm` 不是唯一瓶颈

对 5 条样本做 FP8 scope profiling 后，naive dynamic W8A8 的主要开销为：

| 组件 | CUDA 时间 | 占 measured FP8 |
| --- | ---: | ---: |
| dynamic scale + activation quant prepare | 1363.039 ms | 61.7% |
| `_scaled_mm` | 846.108 ms | 38.3% |
| 合计 | 2209.146 ms | 100.0% |

这里的关键结论是：初始负收益不是因为 `_scaled_mm` 完全没有用，而是因为 dynamic activation quant 的实现太碎，包含 absmax、div、clamp、cast 等多个算子和 kernel launch。对 batch=1、beam=32、短 decode 的生成场景，这类额外 launch 开销非常明显。

换句话说，W8A8 的实际路径变成了：

```text
BF16 Linear:        F.linear / GEMM
naive W8A8 Linear:  absmax + scale + div + clamp + cast + _scaled_mm + bias/reshape
```

如果 activation quant prepare 比 GEMM 本身还贵，那么即使 `_scaled_mm` 对矩阵乘法有加速，端到端也会被量化前处理吞掉。

## 3. 矩阵形状分析：prefill 有收益，decode 小矩阵不合算

进一步捕获实际 Linear 输入形状后，OneRec 生成路径大致分为两类：

- prefill：HF beam search 会把输入扩展到 beam 维，典型形状类似 `(32, seq_len, hidden)`，flatten 后 M 可达到两万级，例如 M≈21632。
- decode：每步只处理新 token，形状类似 `(32, 1, hidden)`，flatten 后 M=32。

虽然矩阵乘法作用在最后一维 hidden 上，但 flatten 后的第一维 M 决定了 GEMM 的 batch/行数规模。M 大时，FP8 `_scaled_mm` 更容易摊薄量化和 launch 成本；M 小时，kernel launch、scale 处理、dtype 转换等固定开销占比更高，W8A8 很难比 BF16 Linear 快。

因此得到两个判断：

1. prefill 的大矩阵是 W8A8 最有可能获得收益的部分。
2. decode 的 M=32 小矩阵不适合强行走 W8A8，至少在当前 PyTorch `_scaled_mm` 路径下不合算。

这解释了为什么只做 naive W8A8 会慢，也解释了为什么单独绕过 decode 只能带来有限收益。

### 3.1 GEMM-only 对比：FP8 乘法在大矩阵上确实更快

为了单独验证 `_scaled_mm` 的矩阵乘法收益，我们又跑了一个 GEMM-only benchmark。这里的 FP8 tensor 和 scale 都提前准备好，计时只包含 BF16 `torch.mm` 或 FP8 `torch._scaled_mm` 本身，不包含 activation dynamic quant、scale 计算、layout 转换、bias、reshape 或 HF generate 开销。设备为 RTX 5090，形状按 OneRec-1.7B 的 hidden/intermediate 维度和实际 beam 展开后的 M 构造。

| 场景 | M | K | N | BF16 mean ms | FP8 mean ms | FP8 GEMM speedup | BF16 TFLOP/s | FP8 TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode q/k/v/o 类 | 32 | 2048 | 2048 | 0.0221 | 0.0291 | 0.76x | 12.17 | 9.22 |
| decode gate/up 类 | 32 | 2048 | 6144 | 0.0372 | 0.0341 | 1.09x | 21.66 | 23.59 |
| decode down 类 | 32 | 6144 | 2048 | 0.0336 | 0.0332 | 1.01x | 23.93 | 24.27 |
| prefill q/k/v/o 类 | 23744 | 2048 | 2048 | 0.8762 | 0.5096 | 1.72x | 227.31 | 390.84 |
| prefill gate/up 类 | 23744 | 2048 | 6144 | 2.6071 | 1.2924 | 2.02x | 229.20 | 462.35 |
| prefill down 类 | 23744 | 6144 | 2048 | 2.5737 | 1.3354 | 1.93x | 232.17 | 447.48 |

这张表把两个现象分开了：

1. 对 prefill 的大矩阵，FP8 `_scaled_mm` 的乘法本身确实明显快于 BF16，速度大约是 1.7x-2.0x。
2. 对 decode 的 M=32 小矩阵，FP8 乘法没有稳定优势，q/k/v/o 形状甚至慢于 BF16。

因此，初始 naive W8A8 的负收益不能解读为“FP8 乘法没有价值”。更准确的判断是：prefill 大矩阵中的 FP8 GEMM 优势被 dynamic activation quant 的额外开销抵消了；decode 小矩阵则本身就不适合强行走 W8A8。后续优化才会分成两条线：prefill 保留 FP8 GEMM 并优化 dynamic quant，decode 则用 A16 路径绕过小矩阵负收益。

## 4. 尝试 static activation：能减少 scale 计算，但不够

为了验证 dynamic scale 计算是否是主要问题，我们尝试了 static activation scale。结果：

| 方法 | generate 总时长 | tokens/s | 相对 BF16 | 相对 naive dynamic |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 541.635 s | 177.241 | 1.000x | - |
| naive W8A8 dynamic | 651.489 s | 147.355 | 0.831x | 1.000x |
| static activation W8A8 | 553.092 s | 173.570 | 0.979x | 1.178x |

static activation 明显改善了初始 dynamic W8A8，但仍没有超过 BF16。profile 也符合这个现象：

| 组件 | naive dynamic | static activation |
| --- | ---: | ---: |
| activation prepare | 1363.039 ms | 797.959 ms |
| `_scaled_mm` | 846.108 ms | 843.803 ms |
| measured FP8 合计 | 2209.146 ms | 1641.763 ms |

static 去掉了 per-token dynamic absmax 的一部分开销，但仍需要 activation quantize，即仍有 FP8 cast/quantization 过程。更重要的是，我们当前不想把主方案切到 static activation，因为推荐模型输入分布随样本和位置变化较大，static scale 可能带来额外精度风险；当前目标是先把 dynamic activation quant 的工程开销降下来。

## 5. decode A16：绕过小矩阵，但单独收益有限

基于 M=32 decode 小矩阵不合算的观察，我们加入了 `--decode_a16_single_token`：

- prefill 仍走 W8A8 `_scaled_mm`。
- decode 的 `seq_len == 1` Linear 改为 BF16 activation + QDQ weight 的 `F.linear`，即 W8A16 模拟路径。

结果如下：

| 方法 | generate 总时长 | tokens/s | 相对 BF16 | 相对 naive dynamic |
| --- | ---: | ---: | ---: | ---: |
| naive W8A8 dynamic | 651.489 s | 147.355 | 0.831x | 1.000x |
| dynamic W8A8 + decode A16 | 634.380 s | 151.329 | 0.854x | 1.027x |

decode A16 有正向作用，但只有约 2.7% 提升。这说明 decode 小矩阵确实是负收益点，但不是最大的瓶颈；最大的瓶颈仍然是 prefill/dynamic activation quant 的实现方式。

## 6. 关键优化：用 vLLM fused dynamic FP8 quant 替换手写 dynamic quant

原始 dynamic activation quant 是用 PyTorch 张量操作拼出来的，大致包含：

```text
absmax -> scale -> div -> clamp -> to(float8)
```

这些操作会产生多个 kernel 和大量 launch overhead。我们参考 vLLM 的 dynamic activation quantization 实现，接入 `vllm._custom_ops.scaled_fp8_quant`，把 per-token dynamic scale 计算和 FP8 quant 合并到一个 fused CUDA op 中。

为了更清楚地说明优化发生在哪里，可以把 dynamic activation quant 的开销单独拆开看。这里使用 5 条样本的 profiler 结果：

| 路径 | profiler scope / op | 调用次数 | CUDA 时间 | 说明 |
| --- | --- | ---: | ---: | --- |
| 优化前 hand-written dynamic | `real_fp8/activation_dynamic_scale` | 1680 | 574.942 ms | per-token absmax 和 scale 计算 |
| 优化前 hand-written dynamic | `real_fp8/activation_quantize` | 1680 | 788.097 ms | `div -> clamp -> to(float8)` 量化路径 |
| 优化前 hand-written dynamic | dynamic activation prepare 合计 | - | 1363.039 ms | 上面两部分相加 |
| 优化后 vLLM fused dynamic | `real_fp8/activation_dynamic_fused_quantize` | 560 | 70.051 ms | scale 计算和 FP8 cast 融合到一个 custom op |

按这个可比口径，dynamic activation quant 从 1363.039 ms 降到 70.051 ms，约为 **19.5x** 加速。这里调用次数也从 1680 次降到 560 次，原因是 fused 路径保留了 q/k/v、gate/up 的 shared input quant，同时每次调用内部不再拆成多个 PyTorch op。

再看 torch profiler 的 top CUDA ops，优化前的 dynamic quant 确实被一串碎算子占据：

| 优化前 top op | 调用次数 | CUDA 时间 | 对应步骤 |
| --- | ---: | ---: | --- |
| `aten::abs` | 3360 | 543.239 ms | `abs(x)`，为 absmax 做准备 |
| `aten::amax` | 1680 | 161.330 ms | per-token absmax reduction |
| `aten::div` | 3390 | 288.064 ms | `x / scale` |
| `aten::clamp` | 3360 | 335.406 ms | clamp 到 FP8 E4M3 可表示范围 |
| `aten::to` / `aten::_to_copy` | 18295 / 8875 | 922.010 ms | dtype/device cast，其中包含 FP8 cast 以及其他路径的 cast |
| `aten::copy_` | 11754 | 926.190 ms | cast/copy 相关开销，和 `to/_to_copy` 有嵌套关系 |

上表中的 `to/_to_copy/copy_` 是全局 torch profiler 统计，包含嵌套和少量非 activation quant 的 cast/copy，所以不应该和其他行直接求和；但它能说明问题形态：原始 dynamic quant 不是一个单独 kernel，而是由多个 PyTorch op 和大量 kernel launch 拼出来。

优化后，相同 profiler 中原来的 `abs/amax/clamp` 已不再出现在 top CUDA ops 里，核心量化路径变成 vLLM fused op：

| 优化后 top op | 调用次数 | CUDA 时间 | 说明 |
| --- | ---: | ---: | --- |
| `_C::dynamic_per_token_scaled_fp8_quant` | 560 | 92.049 ms | vLLM custom op 的 PyTorch 入口统计 |
| `vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided` | 560 | 70.044 ms | 实际 fused CUDA kernel |
| `aten::div` | 30 | 0.047 ms | generate 其他路径残留，已不是 activation quant 主开销 |

接入后，5 条样本的 FP8 scope profile 变为：

| 组件 | CUDA 时间 |
| --- | ---: |
| fused dynamic activation quant | 70.051 ms |
| `_scaled_mm` | 776.863 ms |
| decode W8A16 Linear | 36.486 ms |

和初始 dynamic quant 对比，activation quant 相关开销从 1363.039 ms 降到约 70.051 ms，下降了一个数量级。这是整个探索中最关键的工程优化。

1000 条样本的最终结果：

| 方法 | generate 总时长 | tokens/s | 相对 BF16 | 相对 naive dynamic |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 541.635 s | 177.241 | 1.000x | - |
| naive W8A8 dynamic | 651.489 s | 147.355 | 0.831x | 1.000x |
| vLLM fused dynamic W8A8 + decode A16 | 412.569 s | 232.688 | 1.313x | 1.579x |

此时 W8A8 终于从负收益变成了明显正收益。

## 7. 当前结论

当前探索可以归纳为以下几点：

1. naive W8A8 慢，不代表 FP8 GEMM 没价值。初始慢主要是 dynamic activation quant 的实现开销过大。
2. `_scaled_mm` 在 prefill 大矩阵上有收益，但在 decode 小矩阵上收益不足，甚至可能负收益。
3. dynamic activation quant 必须 fused，否则 abs/div/clamp/to 等碎算子会吞掉 FP8 GEMM 的收益。
4. static activation 能改善时延，但它改变了 activation scale 策略，不是我们当前最希望依赖的主方案。
5. decode A16 是合理的工程补丁，但单独收益有限；它需要和 fused dynamic quant 结合才有明显端到端收益。
6. 当前 real quant 的正收益主要来自两件事：prefill 继续使用 FP8 `_scaled_mm`，dynamic activation quant 改为 fused CUDA op；decode 小矩阵则绕过 W8A8。

## 8. 后续工程方向

短期可继续探索的方向：

- 用 profiler 再确认 fused 版本下剩余瓶颈，尤其是 `_scaled_mm`、HF generate/beam search、KV cache、`aten::to/copy/cat` 等非量化开销。
- 对固定 shape 的 decode 尝试 CUDA Graph 或 `torch.compile`，目标是降低 batch=1 生成中的 kernel launch overhead。
- 如果后续要做真正部署，而不是 PyTorch 模拟，需要评估更成熟的 FP8 Linear kernel 栈，例如 vLLM/flashinfer/cutlass/triton fused linear，而不是长期依赖裸 `torch._scaled_mm` 包装。
- 继续保持算法精度路径和 real quant 路径对齐：例如 GPTQ/weighted GPTQ 属于离线权重量化，不应改变 timed generation 的 runtime kernel 结构。

## 9. 一句话总结

这轮探索的核心结论是：OneRec 的 W8A8 real quant 加速不是简单替换 Linear 为 `_scaled_mm` 就能获得；必须把 dynamic activation quant 做成 fused kernel，并绕过 decode 的小矩阵负收益路径，才能把理论上的 FP8 GEMM 优势转化为端到端时延收益。
