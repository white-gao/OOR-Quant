# 当前最优 W8A8 GPTQ 方案推理收益估算

日期：2026-06-15

## 1. 结论摘要

当前最优版本是 `weighted_gptq_fp8_w8a8_tail1`。已有全量评测结果如下：

| 模型 | run | pass@32 | recall@32 | 当前 fake-quant eval 平均耗时 |
|---|---|---:|---:|---:|
| OneRec-1.7B | `weighted_gptq_fp8_w8a8_tail1_1p7b_ad_calib1024` | 0.214648 | 0.074350 | 0.4730 s/sample |
| OneRec-8B | `weighted_gptq_fp8_w8a8_tail1_8b_ad_calib1024` | 0.260871 | 0.092259 | 1.2670 s/sample |

注意：当前代码是 fake quant，不是真实 FP8 kernel。上表耗时只用于实验记录，不能直接代表部署收益。下面估算按真实部署场景计算：权重和激活可以触发 FP8 / W8A8 kernel，并能吃到低精度访存收益。

在推荐推理场景下，当前方案相对 BF16 全精度模型的合理收益估计是：

| 模型 | 当前方案原样部署估计 | 工程优化较好时的估计 | 权重/带宽节省 |
|---|---:|---:|---:|
| OneRec-1.7B | 约 1.25x-1.50x | 约 1.35x-1.65x | 约 40%-50% |
| OneRec-8B | 约 1.35x-1.60x | 约 1.45x-1.75x | 约 45%-50% |

这里“当前方案原样部署”指 Transformer block Linear 使用 FP8 W8A8，但 `lm_head` 仍未专门优化，且 `tail1` 可能让 decode 阶段 activation 走 BF16 或 mixed path。“工程优化较好”指额外优化 `lm_head`、beam search top-k、shared-input fused FP8 kernel，以及 tail1 mixed kernel。

## 2. 为什么推荐推理场景不能直接套通用 LLM 估计

当前 AD 推荐推理和普通大模型聊天生成有几个关键不同点：

1. 输入 prompt 后会追加 `<|sid_begin|>`。
2. 只生成 3 个 SID token，对应 `s_a/s_b/s_c`。
3. runner 默认配置是：

```text
num_beams = 32
num_return_sequences = 32
max_new_tokens = 3
```

4. 当前生成没有限制到 SID-only vocabulary，仍然是 full vocab logits。OneRec vocab size 是 `176384`。
5. 生成很短，因此 KV cache 长序列访存不是主要瓶颈；beam search、`lm_head` 和 top-k/scoring 的占比会比长文本生成更突出。

因此，通用 LLM 常见的“FP8 decode throughput 接近 2x”不能直接作为端到端收益。我们的场景更接近：

```text
一次较长 prefill
+ 很短的 beam decode
+ 32 beam 的 full-vocab 输出投影和 beam ranking
```

一般情况下，生成 3 个 token 会对应：

```text
1 次 prefill forward，得到第一个 SID token 的 logits
+ 约 2 次增量 decode forward，生成后续 SID token
```

具体实现中，HF generate 的 first-step beam expansion 可能有细节差异，但核心事实不变：decode 很短，beam 宽度很大，且每步都要处理 full-vocab logits。

## 3. 当前最优量化方案做了什么

当前最优版本包含：

- GPTQ-calibrated FP8 weight quantization。
- per-token dynamic activation quantization。
- shared-input activation quantization，例如 Q/K/V 共用一次 activation quant，MLP gate/up 共用一次 activation quant。
- `tail1` activation 保护：prefill 最后一个 token，也就是 `<|sid_begin|>`，以及 decode 阶段 activation 使用 BF16。

所以它不是只做 weight-only，也不是纯 activation quant。更准确地说：

```text
weight side: GPTQ + 手工 token/group 加权
activation side: dynamic per-token quant + shared-input + tail1 BF16 保护
```

## 4. 计算量和访存口径

严格意义上，量化不减少数学 FLOP 数：

```text
Y = XW
```

矩阵乘法维度没有变。节省来自硬件执行成本和访存：

```text
BF16: 2 bytes/element, BF16 Tensor Core
FP8:  1 byte/element, FP8 Tensor Core
```

NVIDIA Hopper 架构文档中，FP8 Tensor Core 相比 FP16/BF16 具有 half data footprint，并且 H100 上 FP8 Tensor Core dense peak 大约是 BF16 的 2x。

参考：
https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

因此更准确的说法是：

```text
FLOP 数不变，但 quantized GEMM 的硬件等效计算时间和权重/激活访存下降。
```

## 5. 模型权重侧估算

本地模型 config：

| 模型 | layers | hidden | intermediate | heads | kv heads | vocab | tie embeddings |
|---|---:|---:|---:|---:|---:|---:|---|
| OneRec-1.7B | 28 | 2048 | 6144 | 16 | 8 | 176384 | true |
| OneRec-8B | 36 | 4096 | 12288 | 32 | 8 | 176384 | false |

每层 Transformer block Linear 近似包括：

```text
attention = q_proj + k_proj + v_proj + o_proj
mlp       = gate_proj + up_proj + down_proj
```

### OneRec-1.7B

```text
Transformer block Linear 参数量: 1.409B
BF16 Transformer Linear 权重访存: 2.82 GB
FP8  Transformer Linear 权重访存: 1.41 GB
Transformer Linear 权重访存节省: 1.41 GB
```

如果把未量化的 `lm_head` 一起计入，整体权重侧访存下降约 `39.8%`。

### OneRec-8B

```text
Transformer block Linear 参数量: 6.946B
BF16 Transformer Linear 权重访存: 13.89 GB
FP8  Transformer Linear 权重访存:  6.95 GB
Transformer Linear 权重访存节省:  6.95 GB
```

如果把未量化的 `lm_head` 一起计入，整体权重侧访存下降约 `45.3%`。

8B 的收益通常会比 1.7B 更明显，因为 Transformer block GEMM 在总耗时里的占比更大，`lm_head` 相对占比更小。

## 6. 推荐/SID/beam search 场景下的收益修正

### 6.1 Prefill 阶段

Prefill 对完整 prompt 做一次前向，prompt 里包含用户历史、文本指令和最后追加的 `<|sid_begin|>`。

这一阶段的特点：

- sequence length 比 decode 长；
- Transformer Linear GEMM 占比较高；
- W8A8 对 q/k/v/o 和 MLP Linear 都有机会加速；
- `tail1` 只保护最后一个 token，理论上大部分 prefill token 仍可走 FP8 activation。

所以 prefill 阶段是当前方案最容易获得接近 FP8 GEMM 收益的部分。

### 6.2 Decode 阶段

Decode 阶段只生成 3 个 SID token，而且使用 32 beam。它和普通长文本 decode 不同：

- 生成步数很短，KV cache 访存不是主要矛盾；
- beam 宽度是 32，decode batch 维度会被 beam 放大；
- 每步都需要 full-vocab logits，vocab size 是 `176384`；
- `tail1` 让 decode activation 保持 BF16，如果部署时直接回退 BF16 GEMM，decode 的 W8A8 收益会下降。

因此 decode 阶段不能简单按“所有 GEMM 都 2x”估算。真正决定 decode 收益的是：

```text
Transformer block 是否有高效 W8A16/FP8-weight mixed kernel
+ lm_head 是否优化
+ full-vocab top-k/beam scoring 是否优化
```

### 6.3 lm_head 和 full-vocab beam 是当前推荐场景的关键扣减项

当前 AD 评测没有把生成限制在合法 SID token 空间，而是 full vocab beam search。也就是说，每个生成 step 都要处理大 vocab logits。

这和很多通用 LLM serving 场景不同。通用聊天模型通常 decode 很长，Transformer block 反复执行，`lm_head` 和 top-k 被更多 token 摊薄；而我们这里只有 3 个 SID token，`lm_head/top-k/beam ranking` 的固定开销更显眼。

所以如果 `lm_head` 不量化、不 fuse、不做 SID candidate 限制，端到端收益会被明显压低。

## 7. 端到端时延估计

用一个简化 Amdahl 估算：

```text
speedup = 1 / ((1 - p) + p / s)
```

其中：

- `p` 是可以被 FP8 W8A8 加速的 Transformer GEMM 占比；
- `s` 是这部分 GEMM 的实际加速，理想接近 2x；
- 非 GEMM、lm_head、beam top-k、tail1 BF16 fallback 不算进 `p`。

在当前推荐场景下，合理判断是：

- 1.7B：非 Transformer GEMM 占比更高，`lm_head` 相对更重，因此端到端收益偏低。
- 8B：Transformer block GEMM 占比更高，因此收益更好。
- 32 beam 会放大 decode 计算，但也放大 full-vocab logits/top-k/scoring 的开销。
- 3-token SID 生成太短，不能像长文本生成那样把固定开销摊薄。

最终估计：

| 部署条件 | OneRec-1.7B | OneRec-8B |
|---|---:|---:|
| Transformer block FP8，`lm_head` 未优化，tail1 decode 可能 BF16 fallback | 1.25x-1.50x | 1.35x-1.60x |
| Transformer block FP8，`lm_head/top-k` 有优化，tail1 有 mixed kernel | 1.35x-1.65x | 1.45x-1.75x |
| 额外做 SID candidate 限制或高效 SID-only decode | 可能继续提升 | 可能接近或略高于 1.8x |

## 8. 可以进一步提高部署收益的方向

| 方向 | 作用 |
|---|---|
| 优化或量化 `lm_head` | 减少 176k vocab 输出投影的瓶颈 |
| SID candidate 限制 / trie constrained decoding | 避免 full-vocab beam search，把推荐场景的结构性先验用起来 |
| fused FP8 Q/K/V、gate/up、down kernels | 提高 shared-input quantization 的实际 kernel 收益 |
| tail1 mixed kernel | 保留 `<|sid_begin|>`/decode BF16 质量收益，同时避免整个 decode 退回 BF16 |
| KV-cache quantization | 对当前 3-token SID 生成帮助有限，但对更长推荐解释/多轮生成有价值 |

## 9. 最终表述建议

不要说：

```text
我们的方案减少了模型 FLOPs。
```

更准确的说法是：

```text
我们的方案不改变矩阵乘法维度，因此数学 FLOP 数基本不变；
收益来自 FP8 Tensor Core 的更高吞吐和 W8A8 的低精度访存。
在 OneRec 推荐场景下，由于只生成 3 个 SID token、使用 32 beam、
且当前仍是 full-vocab beam search，端到端收益会低于理想 2x。
对当前最优的 W8A8 GPTQ+tail1 方案，真实部署时 8B 预计约 1.35x-1.60x，
工程优化较好时约 1.45x-1.75x；1.7B 预计约 1.25x-1.50x，
工程优化较好时约 1.35x-1.65x。
```
