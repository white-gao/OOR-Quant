# Weighted GPTQ 方法总结

本文档总结当前代码中使用的 weighted GPTQ 方法，重点说明它和原始 GPTQ 的关系、token/slot 权重如何得到，以及这些权重如何进入 Hessian。

## 1. 原始 GPTQ 目标

考虑某个 Transformer block 内的一个 Linear 层。设校准集前向时该 Linear 的输入为

$$
X \in \mathbb{R}^{N \times d_{\text{in}}},
$$

其中 \(N\) 是校准样本中所有有效 token 展平后的数量，权重矩阵为

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
$$

原始 GPTQ 使用逐层、逐 Linear 的重建目标：

$$
\min_{\hat W \in \Omega_q}
\left\|XW^\top - X\hat W^\top\right\|_F^2,
$$

其中 \(\Omega_q\) 表示由目标量化格式和 scale 设定共同决定的量化可行集合。也就是说，\(\hat W\) 不是任意实数矩阵，而是量化后可表示的权重矩阵。展开后有

$$
\left\|XW^\top - X\hat W^\top\right\|_F^2
=
\operatorname{Tr}
\left[
(W-\hat W) H (W-\hat W)^\top
\right],
$$

其中

$$
H = X^\top X.
$$

如果把 \(W\) 按 output channel 拆成

$$
W =
\begin{bmatrix}
w_1^\top \\
w_2^\top \\
\cdots \\
w_{d_{\text{out}}}^\top
\end{bmatrix},
$$

则目标等价于

$$
\sum_{j=1}^{d_{\text{out}}}
(w_j-\hat w_j)^\top H (w_j-\hat w_j).
$$

也就是说，原始 GPTQ 中同一个 Linear 的所有 output channel 共享同一个 Hessian \(H\)。

## 2. Weighted GPTQ 的核心改动

当前方法不改变 GPTQ 的权重量化求解框架，而是修改 Hessian 的统计方式。我们认为推荐模型 prompt 中不同 token 对最终推荐目标的重要性不同，因此对不同 token 的重建误差赋予不同权重。

设第 \(i\) 个 token 的权重为

$$
\omega_i \ge 0,
$$

定义对角矩阵

$$
D_\omega = \operatorname{Diag}(\omega_1,\omega_2,\ldots,\omega_N).
$$

weighted GPTQ 的重建目标为

$$
\min_{\hat W \in \Omega_q}
\left\|
D_\omega^{1/2}
\left(
XW^\top - X\hat W^\top
\right)
\right\|_F^2.
$$

展开后得到加权 Hessian：

$$
H_\omega
=
\frac{X^\top D_\omega X}{\sum_{i=1}^{N}\omega_i}.
$$

因此目标可以写为

$$
\sum_{j=1}^{d_{\text{out}}}
(w_j-\hat w_j)^\top H_\omega (w_j-\hat w_j).
$$

这说明当前 weighted GPTQ 仍然是：

- 一个 Linear 一个 Hessian；
- 所有 output channel 共享同一个 \(H_\omega\)；
- 改动发生在 token/sample 维度，而不是 output channel 维度。

对应代码位置：

- `fake_quant_learnable/gptq.py::collect_gptq_hessians`
- 其中 weighted Hessian 通过 `(x2d * weights.unsqueeze(1)).t().matmul(x2d)` 统计。

## 3. SID Slot Token Group

当前推荐模型使用 SID 表示物品。prompt 中的 token 被划分为五类：

$$
\mathcal{G}
=
\{
\text{text},
\text{sid\_a},
\text{sid\_b},
\text{sid\_c},
\text{boundary}
\}.
$$

其中：

- `text`：普通文本 token；
- `sid_a`：SID 第一级 token；
- `sid_b`：SID 第二级 token；
- `sid_c`：SID 第三级 token；
- `boundary`：`<|sid_begin|>` 和 `<|sid_end|>` 等边界 token。

设第 \(i\) 个 token 的 group 为

$$
g(i) \in \mathcal{G}.
$$

最终用于 Hessian 的 token 权重是 group-smoothed 的：

$$
\omega_{l,i}
=
\alpha_{l,g(i)}.
$$

这里 \(l\) 表示 Transformer layer，\(\alpha_{l,g}\) 是第 \(l\) 层中 group \(g\) 的权重。因此当前方法是 layer-wise slot group weighting，而不是每个 token 独立使用一个权重。

对应代码位置：

- `fake_quant_learnable/token_weights.py::build_prompt_slot_token_group_batches`
- `fake_quant_learnable/gradient_weights.py::group_token_weight_batches_by_layer`

## 4. 梯度权重的计算

当前主方法使用 `slot_grad_weighted_gptq`。其权重不是人工固定，而是来自校准集上的推荐目标梯度。

### 4.1 Full-SID Multi-Target Loss

对于一个 prompt \(x\)，其 ground truth 可能包含多个目标 item。每个目标 item 被表示为 SID 序列：

$$
y^{(m)}
=
\left(
y^{(m)}_1,
y^{(m)}_2,
\ldots,
y^{(m)}_T
\right),
$$

在 OneRec 的 SID prediction 中通常对应：

$$
y^{(m)}
=
(\text{sid\_a}, \text{sid\_b}, \text{sid\_c}).
$$

当前 full-SID multi-target 校准 loss 使用 teacher forcing，对最多 \(M\) 个目标 item 计算完整 SID 序列的交叉熵：

$$
\mathcal{L}(x)
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{1}{T}
\sum_{t=1}^{T}
\operatorname{CE}
\left(
p_\theta
\left(
\cdot
\mid
x, y^{(m)}_{<t}
\right),
y^{(m)}_t
\right).
$$

当前实验中常用设置是：

$$
M = 4.
$$

相比只监督第一个 SID token 的 loss，full-SID multi-target loss 更接近实际生成式推荐目标，因为它同时覆盖：

- 多个 ground-truth item；
- 完整 SID 序列；
- `sid_a / sid_b / sid_c` 三个层级。

对应代码位置：

- `real_quant/naive_w8a8/gptq_runtime.py::build_sid_teacher_forcing_target_token_ids`
- `fake_quant_learnable/gradient_weights.py::_teacher_forcing_full_sid_loss`

### 4.2 Token Sensitivity

对第 \(l\) 层输入 hidden state：

$$
H_l^{\text{in}}
\in
\mathbb{R}^{B \times S \times d},
$$

记第 \(i\) 个 token 的 hidden state 为 \(h_{l,i}\)，其梯度为

$$
\frac{\partial \mathcal{L}}{\partial h_{l,i}}.
$$

当前实现使用 activation-gradient product 作为 token sensitivity：

$$
s_{l,i}
=
\frac{1}{d}
\left\|
h_{l,i}
\odot
\frac{\partial \mathcal{L}}{\partial h_{l,i}}
\right\|_1.
$$

直观上，\(s_{l,i}\) 衡量该层中第 \(i\) 个 token 对最终推荐 loss 的局部敏感度。

在 full-SID multi-target 模式下，同一个 prompt 会被展开为多个 teacher-forcing target。实现中会先对多个 target 的 prompt 部分 sensitivity 做平均，再裁剪回原始 prompt token：

$$
s_{l,i}
=
\frac{1}{M}
\sum_{m=1}^{M}
s_{l,i}^{(m)}.
$$

对应代码位置：

- `fake_quant_learnable/gradient_weights.py::collect_gradient_token_weight_batches_by_layer`
- sensitivity 实现为 `(hidden * hidden.grad).abs().mean(dim=-1)`

### 4.3 归一化与截断

得到 token sensitivity 后，代码会做三步处理：

1. 按 percentile 做上界截断；
2. 对有效 token 做 mean normalization；
3. 加入最小 floor，避免部分 token 权重过小。

可以抽象写为：

$$
\tilde{s}_{l,i}
=
\operatorname{clip}
\left(
s_{l,i},
q_p
\right),
$$

其中 \(q_p\) 是有效 token sensitivity 的第 \(p\) 分位数，默认 \(p=99\)。

然后归一化：

$$
\bar{s}_{l,i}
=
\frac{\tilde{s}_{l,i}}
{
\frac{1}{|\mathcal{V}|}
\sum_{k \in \mathcal{V}}
\tilde{s}_{l,k}
},
$$

其中 \(\mathcal{V}\) 是有效 token 集合。最后施加 floor：

$$
\bar{s}_{l,i}
\leftarrow
\max(\bar{s}_{l,i}, \epsilon_{\text{floor}}).
$$

当前默认 floor 为：

$$
\epsilon_{\text{floor}} = 0.05.
$$

对应代码位置：

- `fake_quant_learnable/gradient_weights.py::normalize_gradient_token_weights`

## 5. Slot Group Smoothing

直接使用 token-level 梯度权重可能噪声较大。当前方法会把 token 权重按 SID slot group 聚合：

$$
\alpha_{l,g}
=
\frac{
\sum_{i \in \mathcal{V}}
\mathbf{1}[g(i)=g]\bar{s}_{l,i}
}{
\sum_{i \in \mathcal{V}}
\mathbf{1}[g(i)=g]
}.
$$

然后将同一个 group 内所有 token 的权重替换为该 group 的平均权重：

$$
\omega_{l,i}
=
\alpha_{l,g(i)}.
$$

因此最终 Hessian 是：

$$
H_{l,m}^{\text{slot}}
=
\frac{
X_{l,m}^\top
\operatorname{Diag}
\left(
\omega_l
\right)
X_{l,m}
}{
\sum_i \omega_{l,i}
},
$$

其中：

- \(l\)：Transformer layer；
- \(m\)：该 layer 内的 Linear module；
- \(X_{l,m}\)：该 Linear 的输入 activation；
- \(\omega_l\)：第 \(l\) 层的 slot-smoothed token weights。

注意：同一个 Transformer layer 内的不同 Linear module 使用各自的输入 \(X_{l,m}\)，因此 Hessian 不同；但它们共享同一套 layer-level token weights \(\omega_l\)。

## 6. GPTQ 权重量化过程

得到加权 Hessian 后，后续权重量化仍然沿用 GPTQ 的误差补偿流程。

对一个 Linear weight：

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},
$$

先对 Hessian 做 damping：

$$
H
\leftarrow
H
+
\lambda
\operatorname{mean}
\left(
\operatorname{diag}(H)
\right)
I.
$$

当前默认：

$$
\lambda = 0.01.
$$

然后计算近似逆 Hessian 因子：

$$
U
=
\operatorname{chol}
\left(
H^{-1}
\right).
$$

GPTQ 按 input channel 顺序逐列量化。设当前列为 \(k\)，当前待量化权重列为 \(w_k\)，其 FP8 量化-反量化结果为

$$
q_k = \operatorname{QDQ}_{\mathrm{FP8}}(w_k).
$$

误差项为

$$
e_k
=
\frac{w_k-q_k}{U_{k,k}}.
$$

随后用 Hessian 逆因子对后续列做误差补偿：

$$
W_{:, k+1:}
\leftarrow
W_{:, k+1:}
-
e_k
U_{k,k+1:}.
$$

当前实现按 block 处理，默认 block size 为：

$$
128.
$$

对应代码位置：

- `fake_quant_learnable/gptq.py::gptq_fp8_quantize_weight`

## 7. FP8 权重量化格式

当前权重量化使用 FP8 E4M3 fake quant / real quant 对齐的量化形式。

对每个 output channel \(r\)，使用 per-output-channel scale：

$$
\Delta_r
=
\frac{
\max_k |W_{r,k}|
}{
q_{\max}
}.
$$

然后执行量化-反量化：

$$
\operatorname{QDQ}_{\mathrm{FP8}}(W_{r,k})
=
\Delta_r
\cdot
\operatorname{FP8E4M3}
\left(
\frac{W_{r,k}}{\Delta_r}
\right).
$$

在 fake quant 中保存的是 QDQ 后的 bf16/fp32 权重；在 real quant 中保存的是 FP8 weight 和对应 scale。weighted GPTQ 只影响离线权重量化结果，不增加推理时的 Hessian 计算开销。

## 8. 当前方法的几个变体

当前代码中相关模式可以概括为：

### 8.1 `gptq`

不使用 token 权重：

$$
H = \frac{X^\top X}{N}.
$$

### 8.2 `slot_weighted_gptq`

使用人工设定的 SID slot 权重：

$$
\omega_i = \beta_{g(i)}.
$$

例如可以手动设置：

$$
\beta_{\text{text}},
\beta_{\text{sid\_a}},
\beta_{\text{sid\_b}},
\beta_{\text{sid\_c}},
\beta_{\text{boundary}}.
$$

### 8.3 `slot_grad_weighted_gptq`

使用 full-SID multi-target CE 梯度得到 token sensitivity，再按 SID slot group 聚合：

$$
\omega_{l,i}
=
\alpha_{l,g(i)}.
$$

这是当前主方法。

## 9. 和 GuidedQuant 的区别

GuidedQuant 更细粒度地为不同 output channel 构造不同 Hessian：

$$
H^{(j)}
=
X^\top
\operatorname{Diag}
\left(
\left(
\frac{\partial \mathcal{L}}{\partial Z_{:,j}}
\right)^2
\right)
X.
$$

其中 \(j\) 是 output channel。因此理论上一个 Linear 需要多个 Hessian。

当前 weighted GPTQ 则是：

$$
H_\omega
=
X^\top
\operatorname{Diag}
(\omega)
X.
$$

其中 \(\omega\) 是 token/slot 维度的权重，所有 output channel 共享同一个 Hessian。

两者区别可以总结为：

| 方法 | 权重粒度 | Hessian 个数 | 是否区分 output channel |
| --- | --- | --- | --- |
| GPTQ | 无 token 权重 | 每个 Linear 1 个 | 否 |
| 当前 weighted GPTQ | token / SID slot | 每个 Linear 1 个 | 否 |
| GuidedQuant | token-output-channel element | 每个 channel 或 channel group 1 个 | 是 |

因此当前方法的优势是计算开销和 GPTQ 接近，且能引入推荐模型中的 SID slot 结构；不足是没有建模不同 output channel 对 SID slot 的差异。

## 10. 当前方法的一句话总结

当前 weighted GPTQ 可以概括为：

> 使用推荐校准集上的 full-SID multi-target CE 梯度估计每层中不同 SID slot token 的重要性，将该重要性写入 GPTQ Hessian 的 token/sample 维度，从而让权重量化更优先保持对 SID-based generative recommendation 更关键的 token 重建质量。

