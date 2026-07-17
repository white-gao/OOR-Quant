# Conditional Hessian GPTQ：从 SID Token-Channel 异质性到 GPTAQ 扩展

本文档整理当前 Conditional Hessian PTQ 的研究逻辑、数学目标、已实现版本、初步实验和后续 GPTAQ 扩展。重点是区分以下三件事：

1. 已经由 probe 观测到的事实；
2. 由该事实提出的重建目标；
3. 仍需通过消融实验验证的任务假设。

当前已经实现的是 **hard-routing Conditional GPTQ**。Soft Conditional GPTQ、Conditional GPTAQ 和 stage-aware GPTAQ 仍是后续方案，不应与当前实验结果混为一谈。

## 1. 研究动机

### 1.1 SID 推荐中的 token group

OneRec 的 prompt 同时包含自然语言 token、SID token 和 SID 边界 token。按照 SID 层级，可将 prompt token 划分为五组：

```text
text / boundary / SID-a / SID-b / SID-c
```

记 token group 集合为

$$
\mathcal{G}=\{g_1,g_2,\ldots,g_G\}, \qquad G=5.
$$

这里的分组描述的是 token 在 SID 结构中的角色，而不是视觉/文本模态。SID-a、SID-b 和 SID-c 分别对应层级 SID 的不同 slot。

### 1.2 已有 probe：异质性同时存在于 token 和 channel 两个维度

前期 probe 已从多个角度观测到 SID token 与文本 token、不同 SID slot 之间的分布差异：

- embedding 和 hidden-state 可视化中，SID token 与文本 token 存在明显的表示空间间隔；
- SID-a、SID-b、SID-c 在多层 hidden state 中形成稳定的 slot-specific 聚类；
- token-wise outlier-channel probe 表明，不同 token group 的高响应 channel 集合并不完全一致；
- Conditional Hessian premise probe 表明，同一个 Linear 的不同 output channel 对 token group 的响应分布不同。

均匀抽取 Layer 0、4、8、12、16、20、24、27 后，部分典型结果如下：

| Layer / Linear | 平均归一化熵 | 平均最大 group 概率 | 最大概率不小于 0.35 的 channel 比例 | split-half top-1 一致率 |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 `mlp.gate_proj` | 0.9008 | 0.3599 | 0.4727 | 0.9704 |
| Layer 16 `mlp.down_proj` | 0.9022 | 0.3724 | 0.5674 | 0.9717 |
| Layer 24 `self_attn.q_proj` | 0.9144 | 0.3636 | 0.4805 | 0.9756 |
| Layer 27 `mlp.down_proj` | 0.8300 | 0.4018 | 0.7900 | 0.9771 |

这些结果支持一个比“不同 token group 整体重要性不同”更细的假设：

> token group 的激活几何不仅在 group 之间不同，而且这种差异与 Linear 的 output channel 相关。

但这些 probe 只能证明 **slot-conditioned channel usage** 存在，不能直接证明某个 group 或 channel 对最终推荐指标更重要。

## 2. 记号与普通 GPTQ

考虑一个 Linear：

$$
Y=XW^\top,
$$

其中

$$
X\in\mathbb{R}^{N\times d_{\mathrm{in}}},
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

将第 (r) 个 output channel 对应的权重行记为 (w_r\in\mathbb{R}^{d_{\mathrm{in}}})，量化后的权重行为 \(\widehat w_r\)。定义量化误差

$$
e_r=w_r-\widehat w_r.
$$

普通 GPTQ 的局部重建目标为

$$
\mathcal{L}_{\mathrm{GPTQ}}
=
\frac{1}{N}
\left\|XW^\top-X\widehat W^\top\right\|_F^2.
$$

令

$$
H_{\mathrm{global}}
=
\frac{1}{N}X^\top X,
$$

则

$$
\mathcal{L}_{\mathrm{GPTQ}}
=
\sum_{r=1}^{d_{\mathrm{out}}}
e_r^\top H_{\mathrm{global}}e_r.
$$

因此普通 GPTQ 的关键结构是：

```text
一个 Linear -> 一个 H_global -> 所有 output rows 共享
```

虽然代码通常把所有 output rows 向量化处理，但在上述目标下，不同 output rows 的重建项本身是可分离的。

## 3. Group Hessian 与现有 Weighted GPTQ

设属于 group (g) 的 token 输入构成

$$
X_g\in\mathbb{R}^{N_g\times d_{\mathrm{in}}},
$$

定义 token-count-normalized group Hessian：

$$
H_g
=
\frac{1}{N_g}X_g^\top X_g.
$$

由于各 group 构成 token 集合的一个划分，因此

$$
H_{\mathrm{global}}
=
\sum_{g\in\mathcal{G}}
p_gH_g,
\qquad
p_g=\frac{N_g}{N}.
$$

这说明：仅仅分别计算多个 (H_g)，最后再按 token 数相加，与普通 (X^\top X) 完全等价，并不构成新方法。

现有 group-weighted GPTQ 使用

$$
H_{\beta}
=
\sum_{g\in\mathcal{G}}
\beta_gH_g,
\qquad
\beta_g\ge 0.
$$

其目标为

$$
\mathcal{L}_{\beta}
=
\sum_{r=1}^{d_{\mathrm{out}}}
e_r^\top H_{\beta}e_r.
$$

无论 \(\beta_g\) 来自手工设置还是 end-loss 梯度聚合，同一个 Linear 的所有 output rows 仍共享同一个 Hessian。它只在 token-group 维度重加权，没有显式建模 group 与 output channel 的交互。

## 4. 从 output channel 得到条件分布

### 4.1 Linear 输出能量

对当前 BF16 Linear 输出

$$
Y=XW^\top,
$$

先对每个 token 的输出向量做 RMS 能量归一化。对 token (t) 和 output channel (r)，定义

$$
S_{t,r}
=
\frac{Y_{t,r}^2}
{\frac{1}{d_{\mathrm{out}}}\sum_{k=1}^{d_{\mathrm{out}}}Y_{t,k}^2+\varepsilon}.
$$

该归一化用于削弱不同 token 整体输出幅值的影响，使统计量更关注一个 token 内部的 channel 使用比例。

对 group (g)，计算 channel 条件能量：

$$
A_{g,r}
=
\frac{1}{N_g}
\sum_{t\in g}S_{t,r}.
$$

然后在 group 维度归一化：

$$
\pi_{r,g}
=
\frac{A_{g,r}}
{\sum_{h\in\mathcal{G}}A_{h,r}+\varepsilon}.
$$

对每个 output row (r)，都有

$$
\pi_{r,g}\ge 0,
\qquad
\sum_{g\in\mathcal{G}}\pi_{r,g}=1.
$$

### 4.2 \(\pi_{r,g}\) 的准确含义

\(\pi_{r,g}\) 表示第 (r) 个 output channel 与 token group (g) 的局部响应关联。它不是：

- end-loss 敏感度；
- 推荐指标的重要性；
- 对 token 的因果归因；
- 最优 Hessian 权重的解析解。

它只回答一个更局部的问题：

> 在当前 BF16 Linear 的输出中，第 (r) 个 channel 对哪些 token group 呈现更高的相对响应？

这一解释不要求“输出位置 (t) 只由输入 token (t) 决定”。Transformer hidden state 本来就是上下文聚合结果；这里条件化的是不同 token 位置上的表示分布，而不是把输出严格归因回同位置输入 token。

## 5. Soft Conditional GPTQ 的完整目标

为每个 output row 定义有效 Hessian：

$$
H_r
=
\sum_{g\in\mathcal{G}}
\pi_{r,g}H_g.
$$

对应条件重建目标为

$$
\mathcal{L}_{\mathrm{cond}}
=
\sum_{r=1}^{d_{\mathrm{out}}}
\sum_{g\in\mathcal{G}}
\pi_{r,g}
\frac{1}{N_g}
\left\|X_ge_r\right\|_2^2.
$$

利用 (H_g=N_g^{-1}X_g^\top X_g)，可得

$$
\mathcal{L}_{\mathrm{cond}}
=
\sum_{r=1}^{d_{\mathrm{out}}}
e_r^\top H_re_r.
$$

这给出了 Conditional Hessian 的数学来源：它不是在 GPTQ 更新公式中任意替换矩阵，而是先定义 row-conditioned reconstruction objective，再由该目标得到每个 row 的二阶度量 (H_r)。

因为每个 (H_g) 都是半正定矩阵，且 \(\pi_{r,g}\ge 0\)，所以

$$
H_r\succeq 0.
$$

加入 GPTQ 的标准 diagonal damping 后，可以继续使用 Cholesky 分解和逐列误差补偿。

### 5.1 与已有方法的退化关系

如果对所有 row 都有

$$
\pi_{r,g}=p_g,
$$

则

$$
H_r=H_{\mathrm{global}},
$$

Conditional GPTQ 退化为普通 GPTQ。

如果对所有 row 都有相同但非频率权重

$$
\pi_{r,g}=\beta_g,
$$

则

$$
H_r=H_\beta,
$$

Conditional GPTQ 退化为现有 group-weighted GPTQ。

因此，Conditional GPTQ 的新增自由度不是“有多个 token group”，而是

$$
\pi_{r,g_1}\ne\pi_{s,g_1}
$$

可以对不同 output rows 成立，即 group 权重随 output channel 改变。

## 6. 当前实现：Hard-Routing Conditional GPTQ

当前代码实现的是 soft objective 的最小硬路由近似。对每个 output row 定义

$$
c(r)=\arg\max_{g\in\mathcal{G}}\pi_{r,g}.
$$

然后令

$$
H_r=H_{c(r)}.
$$

实际优化目标为

$$
\mathcal{L}_{\mathrm{hard}}
=
\sum_{r=1}^{d_{\mathrm{out}}}
e_r^\top H_{c(r)}e_r.
$$

### 6.1 为什么可以按 output row 分组执行 GPTQ

在当前 FP8 实现中：

- weight scale 是 per-output-channel；
- 不同 output rows 不共享量化 scale；
- GPTQ 的列补偿发生在同一权重行的 input-feature 维度；
- 重建目标中不同 output rows 相互可分离。

因此可以按 (c(r)) 把 weight rows 分组，对每组权重子矩阵使用对应 (H_g) 独立运行原始 GPTQ。该操作对应上面的 \(\mathcal L_{\mathrm{hard}}\)，不是对代码向量化的近似拼接。

### 6.2 Linear-level 门控

并非所有 Linear 都呈现明显的 channel-conditioned slot structure。当前实现使用归一化熵和高置信 channel 比例决定是否启用 Conditional GPTQ。

对 row (r)，定义归一化熵

$$
h_r
=
-\frac{\sum_{g\in\mathcal G}\pi_{r,g}\log(\pi_{r,g}+\varepsilon)}
{\log G}.
$$

Linear 平均熵为

$$
\bar h
=
\frac{1}{d_{\mathrm{out}}}
\sum_{r=1}^{d_{\mathrm{out}}}h_r.
$$

再定义

$$
q_r=\max_{g\in\mathcal G}\pi_{r,g},
$$

以及超过概率阈值 (q_0) 的 channel 比例

$$
F(q_0)
=
\frac{1}{d_{\mathrm{out}}}
\sum_{r=1}^{d_{\mathrm{out}}}
\mathbf{1}[q_r\ge q_0].
$$

当前默认门控为

$$
\bar h\le 0.94,
\qquad
q_0=0.35,
\qquad
F(q_0)\ge 0.35.
$$

若 Linear 未通过门控，则使用

$$
H_{\mathrm{global}}
=
\sum_g\frac{N_g}{N}H_g
$$

并严格回退到 plain GPTQ。

这些阈值来自当前 probe 的经验分布，不是理论最优值，因此必须通过 threshold ablation 或固定规则跨 domain 验证，不能把调节阈值本身作为主要创新。

### 6.3 部署语义

Conditional GPTQ 只改变离线权重量化：

```text
calibration -> H_g / pi -> row grouping -> GPTQ -> FP8 weight + scale
```

最终导出的仍然是：

```text
FP8 weight + per-output-channel scale
```

推理时不需要 token group、\(\pi\)、多个 Hessian 或动态路由，因此与 plain GPTQ 使用相同的 FP8 kernel，不增加推理时延。

## 7. 当前实现与 Soft Conditional GPTQ 的差异

当前 hard routing 使用

$$
H_r=H_{c(r)},
$$

而完整 soft 版本使用

$$
H_r=\sum_g\pi_{r,g}H_g.
$$

Hard routing 的优点是：

- 每个 Linear 最多需要 (G) 个 Cholesky/GPTQ 路径；
- output rows 可按 group 批量处理；
- 容易验证“group-specific Hessian 是否有用”。

其风险是：

- 丢弃了非主导 group 的信息；
- 当两个 group 概率接近时，\(\arg\max\) 会产生不连续路由；
- 小校准集上的 group 估计误差可能直接改变 row assignment。

Soft Conditional GPTQ 更严格地对应第 5 节的目标，但如果每个 row 都有不同的 (H_r)，逐 row Cholesky 代价过高。更现实的实现是对 \(\pi_{r,:}\) 聚类：

$$
\mathcal C_1,\mathcal C_2,\ldots,\mathcal C_K,
$$

并对 cluster (k) 计算

$$
\bar\pi_{k,g}
=
\frac{1}{|\mathcal C_k|}
\sum_{r\in\mathcal C_k}\pi_{r,g},
$$

$$
H_k
=
\sum_g\bar\pi_{k,g}H_g.
$$

这样只需要 (K) 组分解，同时保留 soft mixture。是否优于 hard routing 需要实验决定。

## 8. 推广到 Activation-Aware GPTAQ

### 8.1 GPTAQ 的非对称重建目标

普通 GPTQ 假设高精度路径和量化路径使用同一个输入 (X)。但真实 W8A8 推理中，量化路径的 Linear 输入还会经过 activation FP8 QDQ。记

$$
X
$$

为 BF16 teacher 输入，记

$$
\widetilde X
$$

为量化路径传播到当前 Linear 后、再经过当前 activation FP8 QDQ 的实际输入。Activation-aware GPTAQ 的目标为

$$
\mathcal L_{\mathrm{GPTAQ}}
=
\left\|XW^\top-\widetilde X\widehat W^\top\right\|_F^2.
$$

对第 (r) 个 output row，仍定义

$$
e_r=w_r-\widehat w_r,
\qquad
\Delta X=X-\widetilde X.
$$

则

$$
Xw_r-\widetilde X\widehat w_r
=
\Delta Xw_r+\widetilde Xe_r.
$$

展开得到

$$
\mathcal L_r
=
e_r^\top H_qe_r
+2w_r^\top De_r
+C_r,
$$

其中

$$
H_q
=
\widetilde X^\top\widetilde X,
$$

$$
D
=
(X-\widetilde X)^\top\widetilde X,
$$

而 (C_r) 与 \(\widehat w_r\) 无关。当前代码中的 `hessian_q` 和 `dxx_t` 分别对应归一化后的 (H_q) 与 (D)。

这说明 GPTAQ 与 GPTQ 的关键差异不是简单地把 (X) 换成 \(\widetilde X\)，而是还存在由输入不对称产生的交叉项 (D)。

### 8.2 Conditional GPTAQ 必须同时条件化 \(H_q\) 和 \(D\)

对每个 token group，定义

$$
H_{q,g}
=
\frac{1}{N_g}
\widetilde X_g^\top\widetilde X_g,
$$

$$
D_g
=
\frac{1}{N_g}
(X_g-\widetilde X_g)^\top\widetilde X_g.
$$

Soft Conditional GPTAQ 对 row (r) 应使用

$$
H_{q,r}
=
\sum_g\pi_{r,g}H_{q,g},
$$

$$
D_r
=
\sum_g\pi_{r,g}D_g.
$$

其目标为

$$
\mathcal L_{\mathrm{cond-AQ}}
=
\sum_r\sum_g\pi_{r,g}
\frac{1}{N_g}
\left\|X_gw_r-\widetilde X_g\widehat w_r\right\|_2^2.
$$

展开后正好得到由 \((H_{q,r},D_r)\) 定义的 row-wise 非对称二次目标。因此：

> Conditional GPTAQ 不能只替换 Hessian 而继续使用全局 (D) 或全局补偿矩阵；这样不再对应上述完整目标。

按当前代码记号，若 (L_r) 是由 (H_{q,r}^{-1}) 得到的上三角 Cholesky factor，则 GPTAQ 的非对称补偿项也必须按 row group 重新构造：

$$
P_r
=
\alpha
\mathrm{triu}(D_rL_r^\top,1)L_r.
$$

Hard Conditional GPTAQ 则令

$$
H_{q,r}=H_{q,c(r)},
\qquad
D_r=D_{c(r)}.
$$

实现时可按 (c(r)) 分组，对每组 weight rows 使用同一组 \((H_{q,g},D_g,P_g)\) 运行现有 GPTAQ。

### 8.3 \(\pi\) 在 GPTAQ 中应保持固定

第一版 Conditional GPTAQ 建议继续从 BF16 Linear 输出计算 \(\pi\)，并在量化优化期间保持不变。这样 \(\pi\) 是校准集上的固定统计量，目标仍是可处理的二次重建目标。

如果让 \(\pi\) 依赖当前量化权重 \(\widehat W\) 或候选量化误差，则目标和 routing 会随优化变量变化，现有 GPTQ/GPTAQ 的闭式补偿推导不再直接成立，需要重新设计交替优化或可学习算法。

## 9. 可考虑的 GPTAQ 变种

### 9.1 Hard Conditional GPTAQ

```text
pi -> argmax group -> (H_q,g, D_g) -> grouped GPTAQ
```

它与当前 hard Conditional GPTQ 最接近，适合作为第一版 GPTAQ 扩展。优点是数学目标明确、实现改动可控；缺点是仍有硬路由不连续问题。

### 9.2 Clustered Soft Conditional GPTAQ

先聚类 output rows 的 \(\pi_{r,:}\)，然后对每个 row cluster 计算

$$
H_{q,k}=\sum_g\bar\pi_{k,g}H_{q,g},
$$

$$
D_k=\sum_g\bar\pi_{k,g}D_g.
$$

每个 cluster 使用自己的 \((H_{q,k},D_k,P_k)\)。这是精度、数学完整性和离线成本之间更现实的折中。

### 9.3 Shrinkage Conditional GPTAQ

当某些 group 的校准 token 数较少时，(H_{q,g}) 可能噪声较大或秩不足。可以向全局统计收缩：

$$
H_{q,g}^{\mathrm{shr}}
=
(1-\rho_g)H_{q,\mathrm{global}}
+\rho_gH_{q,g},
$$

$$
D_g^{\mathrm{shr}}
=
(1-\rho_g)D_{\mathrm{global}}
+\rho_gD_g,
$$

其中 \(0\le\rho_g\le1\)。该变种提高小校准集下的稳定性，但 \(\rho_g\) 的确定方式必须有固定规则或充分消融，否则会引入过多超参数。

### 9.4 Stage-Aware Conditional GPTAQ

当前部署策略可能采用：

```text
prefill: W8A8
decode:  W8A16
```

记 $X_{\mathrm{path}}$ 为量化权重模型传播到当前 Linear、但尚未经过当前 activation quantizer 的输入。此时 prefill 和 decode 使用不同的 activation path：

$$
\widetilde X^{\mathrm{pre}}=Q_{\mathrm{FP8}}(X_{\mathrm{path}}^{\mathrm{pre}}),
$$

$$
\widetilde X^{\mathrm{dec}}=X_{\mathrm{path}}^{\mathrm{dec}}.
$$

这里 decode A16 只表示绕过当前 Linear 的 activation FP8 QDQ，并不表示量化路径输入等于 BF16 teacher 输入；前序权重量化造成的传播误差仍然保留在 $X_{\mathrm{path}}^{\mathrm{dec}}$ 中。

若只使用 prefill W8A8 统计优化 GPTAQ，切换到 decode A16 后可能出现优化目标不一致。更完整的 stage-aware 目标可写为

$$
\mathcal L
=
\lambda_{\mathrm{pre}}\mathcal L_{\mathrm{pre}}
+\lambda_{\mathrm{dec}}\mathcal L_{\mathrm{dec}}.
$$

对应的充分统计量按相同系数组合：

$$
H_{q,r}
=
\lambda_{\mathrm{pre}}H_{q,r}^{\mathrm{pre}}
+\lambda_{\mathrm{dec}}H_{q,r}^{\mathrm{dec}},
$$

$$
D_r
=
\lambda_{\mathrm{pre}}D_r^{\mathrm{pre}}
+\lambda_{\mathrm{dec}}D_r^{\mathrm{dec}}.
$$

该方向需要额外收集 teacher-forcing 或 beam-prefix decode activation，工程和实验复杂度明显更高，不应与第一版 Conditional GPTAQ 同时实现。

## 10. 初步实验结果

当前 `ad / calib128 / test1000` 的初步结果如下：

| 方法 | Pass@16 | Pass@32 | Recall@16 | Recall@32 |
| --- | ---: | ---: | ---: | ---: |
| Plain GPTQ W8A8 | 0.116 | 0.172 | 0.036742 | 0.058266 |
| Conditional GPTQ W8A8 | **0.121** | **0.175** | **0.039110** | **0.060214** |
| Plain GPTQ + decode A16 | **0.126** | **0.185** | **0.041195** | **0.064221** |
| Conditional GPTQ + decode A16 | 0.123 | 0.177 | 0.039546 | 0.060206 |

可以得到两个谨慎结论：

1. 纯 W8A8 下，Conditional GPTQ 在 K=16 和 K=32 上呈现小幅正向结果；
2. 当前 Conditional GPTQ 没有稳定获得 decode A16 的收益。

由于 test1000 上的绝对变化较小，第一点必须通过 `calib1024 + full test` 确认。第二点也不能直接归因于 activation-aware 目标错配，因为当前 Conditional GPTQ 仍是 weight-only GPTQ；beam search 的离散路径变化和 hard routing 的误差形态都可能影响结果。

## 11. 必要的对比与消融

### 11.1 第一阶段：确认主效应

统一使用纯 W8A8：

| 方法 | 目的 |
| --- | --- |
| Plain GPTQ | 基础二阶量化基线 |
| Conditional GPTQ | 验证当前 hard conditional routing 是否有收益 |

先在 `calib128/test1000` 快速筛选，再在 `calib1024/full test` 确认稳定性。

### 11.2 第二阶段：解释收益来源

| 消融 | 定义 | 回答的问题 |
| --- | --- | --- |
| Slot-balanced GPTQ | 所有 row 共享 (G^{-1}\sum_gH_g) | 收益是否只来自 group balance |
| Conditional ungated | 所有 Linear 使用 hard routing | Linear 门控是否必要 |
| Conditional gated | 当前默认方法 | 完整方法 |
| Random routing | 保持各 group row 数不变，随机打乱 row assignment | \(\pi\) 是否提供有效对应关系 |
| Slot-gradient weighted GPTQ | end-loss 梯度得到 group scalar | 局部条件几何与长距离任务梯度的区别 |

最关键的逻辑是：

```text
Slot-balanced > Plain
    说明 group-normalized Hessian 有价值。

Conditional > Slot-balanced
    说明 group × channel 条件化有额外价值。

Conditional > Random routing
    说明 pi 对 row 与 group 的对应不是任意分配。
```

### 11.3 机制指标

除了最终推荐指标，还应报告局部机制指标：

- 每层通过门控的 Linear 数量与名称；
- 不同 Linear 的平均熵和高置信 channel 比例；
- 各 token group 的重建误差；
- overall token-weighted reconstruction error；
- worst-group 或 group macro-average reconstruction error；
- split-half 或不同 calibration subset 下的 row assignment 稳定性。

这些指标用于证明算法确实改变了 slot-conditioned reconstruction geometry，但不应被表述为最终推荐指标的替代品。

### 11.4 最终实验协议

建议区分两个表：

```text
算法消融：所有方法统一 full W8A8
部署结果：所有方法统一 W8A8 prefill + W8A16 decode
```

算法消融不能把 full-W8A8 baseline 与 decode-A16 方法直接比较，否则无法区分离线量化算法和推理策略带来的收益。

最终主实验应覆盖：

- domain：ad / product / video；
- model：OneRec-1.7B，资源允许时补 OneRec-8B；
- metrics：Pass@1/16/32、Recall@1/16/32；
- calibration：固定 1024，必要时补 calibration-size ablation；
- test：各 domain 完整独立测试集。

## 12. 方法边界与当前风险

### 12.1 数学完整不等于任务最优

给定固定的 \(\pi_{r,g}\)，Conditional GPTQ/GPTAQ 都可以写成明确的 row-conditioned reconstruction objective，数学上是完整的。

但这不意味着基于输出能量得到的 \(\pi\) 是推荐任务的最优权重。它是 channel-slot association，不是 end-loss sensitivity。最终是否改善推荐性能只能由实验回答。

### 12.2 Hard routing 可能过强

当前实现把每个 row 完全交给单一 group Hessian，可能牺牲其他 group 的重建。若全量结果不稳定，优先考虑 clustered soft mixture 或 global shrinkage，而不是继续调大量 gate threshold。

### 12.3 Group Hessian 的样本噪声

每个 (H_g) 使用 group 内 token 平均，避免 token 数多的 group天然占据更大权重，但也会放大小 group 的统计噪声。需要检查 condition number、Cholesky jitter 次数以及不同 calibration subset 下的稳定性。

### 12.4 与相关工作的区别需要谨慎表述

- VLM-PTQ 等方法也计算 modality-specific Hessian，但主要将其用于量化参数搜索或 modality-aware calibration；
- GuidedQuant 等方法引入 output-channel-specific 的任务曲率或近似 Hessian；
- 当前方案的特征是利用层级 SID slot 定义 token condition，并通过 output-channel association 构造 row-conditioned GPTQ/GPTAQ reconstruction geometry。

仅替换 token group 名称不足以构成创新。论文贡献必须由 SID slot 结构、row-conditioned 目标、相应消融和跨 domain 结果共同支撑。

## 13. 当前实现与结果位置

核心代码：

- `real_quant/naive_w8a8/conditional_gptq.py`
- `fake_quant_learnable/gptq.py::conditional_gptq_fp8_quantize_weight`
- `real_quant/naive_w8a8/modules.py::RealFP8Linear.from_conditional_gptq_linear`
- `real_quant/naive_w8a8/run_hf_naive_w8a8.py`

Probe：

- `fake_quant_learnable/probe_conditional_hessian.py`
- `fake_quant_learnable/results/analysis/conditional_hessian_probe/ad_1p7b_s128_uniform_l0-27/`

快速实验：

- `real_quant/naive_w8a8/results/ad_1p7b_plain_gptq_w8a8_calib128_test1000/`
- `real_quant/naive_w8a8/results/ad_1p7b_conditional_gptq_w8a8_calib128_test1000/`
- `real_quant/naive_w8a8/results/ad_1p7b_plain_gptq_w8a8_calib128_test1000_decode/`
- `real_quant/naive_w8a8/results/ad_1p7b_conditional_gptq_w8a8_calib128_test1000_decode/`

## 14. 推荐推进顺序

1. 完成 plain GPTQ 与 hard Conditional GPTQ 的 `calib1024/full test` 对比；
2. 若 Conditional GPTQ 收益稳定，补 Slot-balanced 和 Random routing 两个关键消融；
3. 比较 hard routing 与 clustered soft mixture；
4. 在 GPTAQ 中同步实现 group-specific (H_{q,g}) 和 (D_g)；
5. 最后再评估是否需要 stage-aware prefill/decode 目标。

这一顺序可以逐步验证每个新增自由度，避免同时引入 group balance、row routing、activation asymmetry 和 decode stage 后无法解释最终收益来源。
