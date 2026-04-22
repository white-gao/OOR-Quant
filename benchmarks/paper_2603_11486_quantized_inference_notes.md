# `2603.11486v1` 论文笔记：Quantized Inference for OneRec-V2

本文档整理我们对 `2603.11486v1.pdf` 的讨论，重点关注它的论文定位、量化路线，以及我们后续工作的可区分空间。

## 论文定位

这篇论文更适合理解为一篇 **infra / system deployment 论文**，而不是量化算法论文。

它的核心观点是：

> 传统推荐模型由于数值分布和计算形态问题，做低精度推理比较困难；但 OneRec-V2 这类生成式推荐模型在数值分布和计算形态上更接近 LLM，因此 FP8 推理在生产环境中是可行且有效的。

它最强的证据来自工业部署：

- 端到端推理时延下降；
- 吞吐提升；
- 线上 A/B 指标无明显退化；
- 针对 FP8 推理链路做了系统和 kernel 优化。

因此，这篇论文的主要贡献不是提出新的量化算法，而是证明 OneRec-V2 这类生产级生成式推荐模型可以稳定接入 FP8 推理，并获得实际系统收益。

## 量化路线

从论文描述看，它使用的是比较标准的 scale-based FP8 PTQ。

大致路线如下：

- 数值格式：FP8 E4M3；
- 量化对象：
  - Attention 中的 `q/k/v/o` Linear；
  - Dense FFN 中的 Linear；
  - Sparse MoE 的 grouped GEMM；
- weight：
  - 离线量化；
  - Linear weight 使用 per-channel scale；
  - 以 FP8 weight + FP32 scale 的形式存储；
- activation：
  - 推理时动态量化；
  - 按 token 计算 runtime scale；
- 计算：
  - FP8 TensorCore matmul；
  - FP32 accumulation；
  - matmul 输出再 cast 回 FP16；
- 其他模块：
  - 对数值敏感或计算占比不高的模块保留 FP16。

论文中给出的量化公式是：

```text
x_hat = Q(x; s) = round(x / s)
```

因此，从算法角度看，它基本可以理解为 **带 scale 的 Round-to-Nearest PTQ**。论文没有描述 GPTQ、AWQ、SmoothQuant、Hessian compensation、activation-aware weight rescaling、learned scale 或 ranking-aware calibration 这类方法。

严格来说，论文没有展开底层硬件 rounding mode 的实现细节，所以不能断言它的 kernel 一定是最朴素的 RTN；但从算法描述上看，它就是标准 scale-based PTQ，而不是新的量化算法。

## MoE 相关部分

这篇论文和我们当前 OneRec-1.7B 实验的一个重要区别是：它研究的是生产级 OneRec-V2 fat-MoE 模型。

由于模型里有 Sparse MoE，因此它需要专门处理 MoE grouped GEMM：

- activation block granularity：`1 x 128`；
- weight block granularity：`128 x 128`；
- grouped GEMM 执行；
- Hopper TMA-enabled kernels；
- operator fusion；
- TensorCore 利用率优化。

这确实是它和我们当前 dense OneRec-1.7B 设置的一个明显区别。但这个区别主要是工程和系统层面的，不是量化算法层面的根本创新。

## 论文贡献拆解

可以把这篇论文的贡献分成三层。

算法层面：

- 使用标准 FP8 PTQ；
- 没有明显新的量化 objective 或优化方法；
- 没有和 GPTQ、AWQ、SmoothQuant 等通用 PTQ 方法做细粒度对比。

模型 / workload 层面：

- 说明 OneRec-V2 的 weight 和 activation 分布比传统推荐模型更接近 LLM；
- 论证生成式推荐模型比传统推荐 pipeline 更适合低精度推理；
- 由于 OneRec-V2 是 fat-MoE 架构，因此处理了 MoE grouped GEMM 的量化和执行。

系统层面：

- 将 FP8 推理接入生产 serving stack；
- 使用 TensorRT / 自研基础设施；
- 优化 FP8 GEMM、TopK、attention 和 MoE execution；
- 报告真实 latency、throughput 和线上 A/B 结果。

其中系统层面的贡献最强。

## 和我们当前工作的区别

我们当前工作和它有几处明显不同：

- 我们研究的是 open OneRec-1.7B dense 模型，不是生产级 OneRec-V2 MoE 模型；
- 我们目前做的是 FP8 E4M3 weight-only QDQ simulation：
  - BF16 weight -> FP8 quantize -> BF16 dequantize；
  - vLLM 仍然走 BF16 计算；
  - 目前没有真实 FP8 TensorCore 加速；
- 我们重点关注推荐评估行为：
  - `pass@k`；
  - `recall@k`；
  - PID-level metrics；
  - beam ranking perturbation；
  - top-k candidate stability；
- 我们还没有实现 activation quantization 或真实低精度 kernel。

所以目前最准确的区别是：

> 这篇论文研究的是生产级 OneRec-V2 MoE 的 FP8 W8A8 系统部署；我们当前研究的是开源生成式推荐 dense 模型在 QDQ simulation 下的量化行为和推荐 ranking 指标扰动。

这个区别是存在的，但如果要形成算法型论文，我们还需要找到更清晰的研究问题。

## 不适合作为我们主创新的点

我们后续不应该把下面这些点作为主要 novelty：

- “FP8 quantization 对生成式推荐是可行的”；
- “OneRec-like 模型的数值分布比较可控”；
- “生成式推荐模型比传统推荐模型更接近 LLM”。

这些观点已经被该论文用更强的工业证据讲过了。

我们也不应该把简单的 FP8 QDQ 结果包装成独立贡献，否则容易看起来只是对该论文 FP8 可行性结论的开源复现。

## 可能的研究空白

这篇论文仍然留下了一些更偏 algorithm / empirical research 的空间。

### 1. 推荐指标下的量化敏感性

论文没有细致分析量化如何影响推荐 top-k 指标。

可以继续研究的问题包括：

- 哪些 layer / module 对推荐指标最敏感？
- weight reconstruction error 小，是否一定意味着 `recall@k` / `pass@k` 稳定？
- 推荐指标的下降是否和 tensor-level MSE 不一致？
- 量化主要是在扰动 top-k ranking boundary，还是破坏候选生成能力？

这是目前最值得我们继续推进的方向。

### 2. Ranking-Aware Quantization

论文没有提出和推荐 ranking objective 对齐的量化方法。

可能的方向是：

- 用少量 calibration prompts 统计 baseline 和 quantized 模型的 top-k ranking 差异；
- 找出对推荐指标扰动大的 layer / module；
- 对敏感模块保留高精度；
- 对不敏感模块更激进量化；
- 在相同压缩比例下，对比 uniform FP8、naive QDQ、AWQ/GPTQ、random mixed precision。

这样问题就从：

> FP8 能不能跑？

变成：

> 在推荐 top-k 目标下，精度应该如何分配？

### 3. 更低 bit 的量化边界

论文的 limitation 里明确说没有探索 INT8、FP6、FP4、INT4 或 mixed lower-bit schemes。

这给我们留下一个可能空间：

- FP8 对 OneRec-like 模型可能太容易；
- 真正有算法问题的地方可能在 INT4 / FP4 / mixed precision；
- 可以研究推荐指标下的 accuracy-compression frontier。

这个方向比继续做 FP8 weight-only 更可能暴露真实量化问题。

### 4. 开源可复现 Benchmark

该论文基于生产系统，外部很难复现。

我们可以考虑构建开源生成式推荐模型上的可复现实验：

- 多 domain；
- 多模型规模；
- 多种量化方案；
- 推荐特定指标；
- layer/module sensitivity；
- beam ranking perturbation。

不过如果只有 benchmark，贡献可能偏 empirical；如果想更像算法论文，最好再配合一个方法。

### 5. Activation Quantization 分析

论文使用了 runtime per-token activation quantization，但没有深入分析推荐 decoding 位置上的 activation 行为。

可能研究的问题包括：

- `<|sid_begin|>` 和生成 SID 位置的 activation 分布；
- stable / lost / gained / boundary 样本之间是否有 activation 差异；
- activation quantization 是否有推荐特定 failure mode。

不过我们目前单样本 profiling 没有看到明显的 SID-vs-text activation 本质差异，所以这个方向暂时应该保持探索态度，不适合作为主线押注。

## 建议的下一步

目前最实际的下一步不是重复这篇论文的 FP8 可行性故事，而是研究：

> 推荐 top-k 指标下的量化敏感性。

具体可以先做：

1. 构建 layer/module sensitivity map：
   - 每次只量化一个 layer 或 module group；
   - 在 `ad_sample_1000` 上评估；
   - 记录推荐指标和 beam ranking perturbation。

2. 对比不同误差信号：
   - weight MSE；
   - activation / output MSE；
   - `pass@k` / `recall@k` drop；
   - top32 overlap；
   - top1 change rate。

3. 判断 reconstruction error 和 recommendation metric drop 是否不一致。

4. 如果存在明显不一致，再设计 sensitivity-aware mixed precision：
   - 高敏感模块保留高精度；
   - 低敏感模块更激进量化；
   - 与 uniform FP8 和通用量化 baseline 对比。

这样可以形成更清楚的区别：

> 现有论文证明 FP8 inference 可以在工业 OneRec-V2 系统中高效部署；我们的工作研究开源生成式推荐模型中，量化如何影响推荐 ranking 行为，并尝试设计与 top-k 推荐指标对齐的精度分配策略。

## 当前结论

这篇论文可以作为背景文献引用，用来说明 FP8 inference 在生成式推荐场景下已经被证明具有工业可行性。

但它没有解决生成式推荐模型在推荐 ranking objective 下应该如何量化的问题。这个问题仍然有可能成为我们和它区分的研究空间。
