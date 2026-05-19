# Ranking-Margin SmoothQuant

这个目录实现一个面向推荐指标的 SmoothQuant 变体。目标不是对 `alpha` 做普通网格搜索，而是把 Recall@K 相关的排序边界信息转成 channel importance，再用于修正 SmoothQuant 的 per-channel scale。

## Motivation

vanilla SmoothQuant 的 scale计算：

$$
s_c = \frac{a_c^\alpha}{w_c^{1-\alpha}}
$$

其中 $a_c$ 是 calibration activation absmax，$w_c$ 是对应 Linear weight 输入 channel 的 absmax。这个公式优化的是低精度矩阵乘的数值范围分配，但没有显式关注推荐任务中的 top-k 排序边界。

推荐 SID prediction 的核心风险是：量化扰动让正样本 SID 从 top-k 边界内掉出去。因此这里用 margin surrogate 估计“哪些 channel 对排序边界更敏感”。

## Objective

对 calibration 样本 $i$，设正 SID token 的 logit 为 $z_i^+$，第 $K$（这里K就是num_beams） 个 hard negative token 的 logit 为 $z_i^-$，margin 为：

$$
m_i = z_i^+ - z_i^-
$$

使用一个 ranking loss：

$$
\mathcal{L}_{\text{rank}}
= \frac{1}{N}\sum_i \eta_i \cdot \operatorname{softplus}(\tau - m_i)
$$

$$
\eta_i = \frac{1}{|m_i| + \epsilon}
$$

$\eta_i$ 会放大靠近边界的样本；这些样本更可能影响 Recall@K。

对某个 Linear 输入 activation $x$，用一阶近似估计 channel 重要性：

$$
I_c
= \mathbb{E}_i\left[
\operatorname{mean}_{t}
\left|
x_{i,t,c}\cdot
\frac{\partial \mathcal{L}_{\text{rank}}}{\partial x_{i,t,c}}
\right|
\right]
$$

如果某个 channel 的 $I_c$ 大，说明这个 channel 的扰动更容易改变排序边界，应当在 smoothing 时更偏向保护 activation。

## Scale Formula

从加权 trade-off 角度，可以写成：

$$
\min_{s_c}
\quad
I_{\text{act},c}\left(\frac{a_c}{s_c}\right)^2
+
I_{\text{w},c}\left(w_c s_c\right)^2
$$

如果只保留一个任务相关 importance，并兼容 SmoothQuant，实际实现采用：

$$
\text{base}_c = \frac{a_c^\alpha}{w_c^{1-\alpha}}
$$

$$
g_c =
\operatorname{clip}
\left(
\frac{I_c}{\operatorname{geomean}(I)},
g_{\min},
g_{\max}
\right)
$$

$$
s_c = \text{base}_c \cdot g_c^\beta
$$

其中 $\beta=0$ 时退化为普通 SmoothQuant。$\beta>0$ 时，高 ranking importance channel 的 scale 变大，activation 会被更强地除小，weight 对应列会被更强地放大。

## Result

### Setting

以下结果均基于 AD domain `sample_size=1000`，评测样本为 test split 前 1000 条；calibration / importance 使用同一 test split 中 offset 1000 之后的 128 条样本，避免与 sample1000 测试集重叠。

### Ranking-Margin SmoothQuant

因为想看一下layer间对平滑scale的敏感度，也做了浅层和后层的消融

| method | pass@1 | pass@32 | recall@1 | recall@32 | pid_pass@32 | pid_recall@32 |
|---|---:|---:|---:|---:|---:|---:|
| Plain FP8 W+A | 0.024 | 0.234 | 0.0079 | 0.0829 | 0.222 | 0.0783 |
| SmoothQuant | 0.023 | 0.233 | 0.0065 | 0.0864 | 0.221 | 0.0818 |
| RankingMargin all | 0.023 | 0.231 | 0.0064 | 0.0833 | 0.216 | 0.0786 |
| RankingMargin layer<20 | 0.020 | 0.227 | 0.0052 | 0.0801 | 0.213 | 0.0755 |
| RankingMargin layer>=20 | 0.027 | 0.234 | 0.0086 | 0.0848 | 0.219 | 0.0799 |

结论：

- 当前 Ranking-Margin scale 没有稳定超过 vanilla SmoothQuant。全层 Ranking-Margin 在 `recall@32`、`pid_recall@32` 上均低于 SmoothQuant。
- 只在前 20 层使用 Ranking-Margin 掉点最明显，说明低/中层的 ranking-aware correction 没有带来收益，反而扰动了原本较稳定的 SmoothQuant scale。
- 只在后层使用 Ranking-Margin 能提升 `pass@1/pass@32`，但 `recall@32` 和 `pid_recall@32` 仍低于 SmoothQuant。这更像是高层 scale 扰动造成的 ranking reshuffle，而不是稳定的推荐精度提升。

### Related SmoothQuant Layer Ablation

因为观察到layer间scale敏感度不同，也对smoothquant做了layer浅层和深层的消融

| method | pass@1 | pass@32 | recall@1 | recall@32 | pid_pass@32 | pid_recall@32 |
|---|---:|---:|---:|---:|---:|---:|
| Plain FP8 W+A | 0.024 | 0.234 | 0.0079 | 0.0829 | 0.222 | 0.0783 |
| SmoothQuant all layers | 0.023 | 0.233 | 0.0065 | 0.0864 | 0.221 | 0.0818 |
| SmoothQuant layer<20 | 0.027 | 0.228 | 0.0084 | 0.0819 | 0.214 | 0.0769 |
| SmoothQuant layer>=20 | 0.022 | 0.237 | 0.0062 | 0.0848 | 0.225 | 0.0807 |

结论：

- **SmoothQuant 的主要收益来自模型高层**。只对前 20 层使用 SmoothQuant 是负优化；只对后 8 层使用 SmoothQuant 已经接近 full SmoothQuant，并在 `pass@32/pid_pass@32` 上更高。
- 高层虽然 activation channel 稳定性下降，但仍存在显著的大激活值方向。固定 calibration scale 不是逐样本最优，但仍能捕获一部分稳定 outlier，并降低 activation 量化压力。
- 因此后续不应移除高层 SmoothQuant；更合理的基线是保留 full SmoothQuant，再探索其它误差控制策略。

### Channel Stability

为验证 SmoothQuant 的离线固定 scale 是否合理，这里只看 activation outlier channel 的跨样本稳定性：

```text
activation outlier: A_c = mean |x_c|
```

以下表格使用 top-5% channel 的平均 Jaccard。

| node | activation J@5% |
|---|---:|
| L0.attn_qkv_input | 0.845 |
| L0.ffn_gate_up_input | 0.803 |
| L7.attn_qkv_input | 0.746 |
| L7.ffn_gate_up_input | 0.668 |
| L14.attn_qkv_input | 0.775 |
| L14.ffn_gate_up_input | 0.657 |
| L21.attn_qkv_input | 0.745 |
| L21.ffn_gate_up_input | 0.613 |
| L27.attn_qkv_input | 0.660 |
| L27.ffn_gate_up_input | 0.584 |

按层平均：

| layer | activation J@5% |
|---|---:|
| 0 | 0.824 |
| 7 | 0.707 |
| 14 | 0.716 |
| 21 | 0.679 |
| 27 | 0.622 |

观察：

- Activation outlier channel 在不同样本之间具有较高重叠度，底层最稳定，高层有所下降但仍明显不是随机分布。
- 这说明 OneRec 的大激活值方向具有一定跨样本共性，使用 calibration 样本计算固定 SmoothQuant scale 是合理的。
- 高层 activation outlier 稳定性下降，按理来说越到高层smoothquant的scale越不准；但高层仍存在可复用的大激活值方向，因此高层 SmoothQuant 仍然能带来主要收益。

### Current Takeaway

当前 Ranking-Margin SmoothQuant 是一个相对错误的尝试：

```text
ranking loss 能提供任务相关信号，但直接把 |x * grad| 注入 SmoothQuant scale 并不能稳定提升推荐精度。
```

- 主流 PTQ 优化方法的直接目标通常仍是量化误差或重构误差，例如 weight/activation 的 MSE、layer output reconstruction error 等，任务指标更多是间接受益。Ranking-Margin 方法尝试把推荐排序边界信号注入 SmoothQuant scale，相当于用 task-aware importance 改变量化误差的分配方式，是一个比较激进的尝试。从实验看，直接用 ranking loss 修正 scale 并不稳定，后续更适合回到量化损失本身，在量化误差建模中引入推荐场景约束，而不是直接替代量化目标。
- 校准集通常规模较小，主要用于估计整体 activation 分布，从而计算相对稳定的离线量化参数。在 Ranking-Margin 方法下，小校准集提供的推荐信号非常稀疏，只覆盖有限的正 SID、hard negative 和局部排序边界，难以稳定代表完整测试集的推荐分布。因此基于小校准集学习固定 ranking-aware scale 容易过拟合局部 ranking 信号，泛化性不足。而且现在公式设计上推荐信号比较粗糙。

