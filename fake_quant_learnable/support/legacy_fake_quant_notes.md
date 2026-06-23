# Legacy fake_quant Notes

`fake_quant/` was removed from the active repository tree on 2026-06-17. The current W8A8/GPTQ/token-weight/tail1 implementation is under `fake_quant_learnable/`.

The SmoothQuant helper functions that were still required by active code were migrated to `fake_quant_learnable/support/smoothquant_core.py`. Historical generated outputs under `fake_quant/results/` and `fake_quant/probes/activation_probe/activation_profiles/` were intentionally not kept.

## Removal Record

`fake_quant/` is kept for historical HF fake-quant baselines, SmoothQuant experiments, ranking-margin probes, and activation/weight probe scripts. The active implementation path is `fake_quant_learnable/`.

## Preserved in active code

The SmoothQuant helpers that were still used by the active runner were migrated to `fake_quant_learnable/support/smoothquant_core.py`:

- `compute_smooth_scale`
- `smooth_linear_weight`

Active imports now go through `fake_quant_learnable`, so current W8A8/GPTQ/token-weight/tail1 experiments do not depend on the legacy `fake_quant` package.

## Cleaned on 2026-06-17

The following generated artifacts were removed from the active tree to keep the repository lightweight:

- `fake_quant/results/`
- `fake_quant/probes/activation_probe/activation_profiles/`
- `fake_quant/**/__pycache__/`

The scripts and README commands remain if historical outputs need to be regenerated.

## Historical MAIN.md

本md文档用于记录OneRec量化算法设计的全流程，方便迭代版本管理
## Fake quant
当前市面上的量化算法主要分为基于大模型的传统派，和将大模型量化算法迁移到其他模型的“邪修”派。老油条都知道把别的领域的方法拿到自己领域来做适配就能水一篇论文，那么“邪修”派其实也是这个逻辑。目前大模型量化算法迁移这一方向主要集中在CV模型，例如ViT，DiT，Diffusion Model，ARVR，Mamba等CV领域的模型，都有人进行了量化算法的定制。他们无疑都遵循了同一种思路：

**"CV模型架构和LLM架构有比较大的区别->架构的区别导致模型参数和计算激活值的数值分布和LLM模型有较大区别->数据分布区别导致LLM量化算法无法用于CV模型->针对CV模型的特殊分布来设计专门的量化算法"**

对于我们当前的任务，有以下statement：
- 选用的量化模型为OpenOneRec开源的OneRec系列模型，有四个版本1.7B/1.7B-Pro/8B/8B-Pro，目前用OneRec-1.7B来尝试进行伪量化的尝试
- benchmark选用OpenRecRec开源的benchmark，有多个task，针对当前的业务场景，主要关注item prediction，也就是集中在ad，video，product三个task
- 目前量化的实现方法为“伪量化/fake quant.”，本质上是在推理forward过程中劫持模型layer并进行替换，劫持操作可以将模型参数量化为低精度版本（weight quant），同时也可以将layer输入x量化为低精度版本（activation quant）
- 目前的量化目标当前模型离线存储数值类型为bf16，目标量化到fp8（暂定为E4M3格式），相关资料表明，Qwen3 bf16->fp8的量化推理速度能提升一倍左右，显存可减少70%左右
- 经过领域的不断探索，目前比较有效的量化流程有两个阶段，第一个阶段是参数平滑，算法会在通道维度上计算一个平滑系数，用于平滑异常的参数通道，这个操作是计算等价的，不会造成量化损失，而是为了降低量化的难度；第二个阶段是量化方法，这部分真正进行了数据类型的转换，该阶段设计量化粒度，缩放因子的计算，校准策略，这里是真正产生量化损失的环节

## 量化策略
所有的量化层劫持实现都在quant.py中，目前实现的量化层/平滑算法包括以下内容：
1. fp8_weight_per_channel():
    per-channel版本的模型权重量化，在模型读取截断一次性替换模型参数数值
2. fp8_activation_per_token()：
    per-token版本的激活值动态量化，也就是根据输入的激活值，当场计算scale进行量化
3. fp8_activation_static_tensor():
    per-tensor版本激活值静态量化，静态版本无法实现动态情况下per-token scale的计算

## 平滑策略
1. Smoothquant：
    在模型量化前先利用calibration set为每个channel提前计算平滑scale，再在平滑后的weight和activation上进行量化
2. ranking_margin + smoothquant:
    通过对排序loss求一阶导得到和推荐结果相关的每个channel的重要性分数，分数越大说明这个channel的扰动更容易改变排序边界，应当在smoothing的时候更偏向保护activation，也就是要放大对应channel的scale
    最终是全面掉点，同时还尝试了在浅层和深层使用这个增强平滑的方法，发现在浅层掉点更明显，深层和smoothquant接近。这说明smoothquant本身对浅层的scale计算是比较准确的，再加上ranking就纯负面，而深层smoothquant的scale本身就没那么准，所以再加扰动也不会掉很多点

v1的实现是纯数值模拟，虽然有用上`torch.float8_e4m3fn`，但本质上只是当成个“筛子”，来做数值上的转换，模型的所有计算都是在bf16来做的
v2的实现真正使用了fp8 tensor（既然5090和H100支持，那为什么不用呢？），具体而言就是量化tensor的类型就保持`torch.float8_e4m3fn`，在部分运算上，使用`torch._scaled_mm`来进行fp8 tensor之间的矩阵计算

### weight quant
weight是经过漫长的模型训练的，因为细粒度的梯度下降和正则化，weight整体的分布会一直被往0附近压，所以权重的数值分布一般来说是非常紧凑，平滑，整体呈现正态分布。由于权重天生的偏正态分布，所以一般来说可以直接使用min-max方法来量化，不会有太明显的异常值来拉伸量化区间。

目前已经对模型权重分布进行统计，结果在 `fake_quant/probes/weight_probe` 中，所有线性层（attn qkvo，mlp gate/up/down）权重分布均类似正态分布，数值聚集在±0.1之间。各个layernorm参数分布有所不同，普遍单峰会出现大值，但是一般情况下不对其进行量化。

### activation quant
激活值会受到不同输入数据，条件注入等因素影响，通常来说会存在相比于平均值大数倍甚至数十倍的异常值，而且常常会呈现channel-wise或者token-wise的整体异常情况。
- 从channel/维度的层面来解释，模型channel维度一般代表一种语义，如果token/整条prompt命中了某一channel的语义，那一个channel就会呈现大值；又有另外一种理论是模型自己学到的保留高信息信号的方法
- 从token层面来解释，一般来说是由attention运算机制决定的，attention 权重是per-token的，有的时候score分配不出去就会堆到某一个锚点token，导致token-wise的激活值偏大（存疑），位置编码也是token-wise加的
对于activation的量化来说，由于LLM场景下输入的不确定性，如果要保精度的话，一般是采用在线动态量化策略，如果你的分布可以比较好地预测，也可以使用离线计算scale，然后线上直接量化的策略。而量化的方法有些论文更倾向于采用percentile方法来适配activation存在大异常值的场景，来保大多数数值量化精度损失

目前已经对激活值进行初步的探测，激活值探测开销比较大，所以还没有形成统一的结论。
- 32个sample来统计不同token的激活值分布，结果在 `fake_quant/probes/activation_probe/activation_profiles/v1.0/OneRec-1.7B-ad-sample-32/plots`，不同token的分布基本都保持一直，随着层数逐渐升高，分布曲线逐渐往高值扩张，同时chat_special token出现异常高单值
- 对单个样本来细致查看各层激活值，发现在channel维度上更容易出现普遍异常值现象，token维度上比较难观察到，结果在 `fake_quant/probes/activation_probe/activation_profiles/v1.0/token_channel_sample_0`；对每层异常值channel进行统计，查看异常通道是否存在重叠现象，对该样本的统计发现对于attn和ffn前的异常值channel一般都重叠，而内部attn/ffn内部的线性层（attn o_proj/mlp down_proj）就不太重叠。但是这块还需要再进行拓展实验，一个是样本数量，不同domain样本；另外一个是token数量，目前只看了最后的100多个token，还没看到开头的sink token
- 对不同输入样本激活值通道进行探测，发现样本之间存在较大比例的重叠的channel，随着模型层数的增高比例从80%逐渐降到50%，一定程度上可以作证smoothquant在高层的scale其实是一个方差比较大的平均。

## baseline
- OneRec-V2：offline weight_linear_fp8 + activation_dynamic_fp8，MoE GEMM fp8，数值敏感或者收益低的模块保留fp16（文中没提到是哪里）
  - 仅对计算最密集的算子应用量化，包含 Attention 模块中的 qkvo 投影层、Dense FFN 中的线性变换（Linear 层），以及 Sparse MoE 中的分组 GEMM 操作
  - 对于其他对数值更敏感或计算占比不高的组件，模型依然保持原始的 FP16 精度运行，以降低数值风险
  - 权重 (Weights)：离线计算缩放因子，采用按通道（by channel）的粒度量化为 FP8 。在 GPU 显存中，这些权重以 (FP8 权重, FP32 缩放因子) 的组合形式存储
  - 激活值 (Activations)：在推理运行时，采用按 Token（by token）的粒度动态计算缩放因子，并量化为 FP8
  - 乘法与累加 (MatMul)：矩阵乘法阶段使用 FP8 TensorCore 进行乘法运算，但为了保证精度，累加操作（Accumulation）是在 FP32 精度下进行的乘法与累加 (MatMul)：矩阵乘法阶段使用 FP8 TensorCore 进行乘法运算，但为了保证精度，累加操作（Accumulation）是在 FP32 精度下进行的
  - 针对 MoE（混合专家）模块中的分组 GEMM 操作，为了适应其特殊的路由和并行执行结构，论文采用了分块（block-wise）量化设计：激活值的量化粒度：在张量的最后一个维度上，按照 1x128 的块大小进行量化；权重的量化粒度：按照 128x128 的块大小进行量化

### AD full fake-quant baseline

当前先在 OneRec-1.7B 的 HF fake-quant 路径上跑 AD full，样本数为 27677，beam 设置为 `num_beams=32`、`num_return_sequences=32`、`max_new_tokens=3`。这里的耗时是 HF/PyTorch fake quant 路径的 wall-clock time，只用于记录实验开销，不代表真实 FP8 kernel 加速。

| 实验 | quant 配置 | pass@1 | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | time(min) | avg(s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HF baseline | BF16，无 fake quant | 0.0193 | 0.2166 | 0.0750 | 0.2024 | 0.0695 | 120.94 | 0.2622 |
| FP8 weight-only | Linear weight FP8 per-channel，activation BF16 | 0.0199 | 0.2112 | 0.0729 | 0.1973 | 0.0674 | 120.86 | 0.2620 |
| FP8 weight+act | Linear weight FP8 per-channel，activation FP8 per-token dynamic(shared-input) | 0.0196 | 0.2040 | 0.0701 | 0.1902 | 0.0648 | 212.64 | 0.4610 |
| SmoothQuant & FP8 weight+act |Linear weight FP8 per-channel，activation FP8 per-token dynamic(shared-input) | 0.0182 | 0.2053 | 0.0710 | 0.1911 | 0.0659 | 421 | 0.9180 |

相对 baseline，weight-only 的 `pid_recall@32` 下降约 0.0021，weight+act 的 `pid_recall@32` 下降约 0.0047。初步看 weight FP8 本身不是主要难点，activation dynamic fake quant 带来的推荐指标扰动更明显；同时 fake quant 的 activation QDQ 会显著增加 PyTorch 路径耗时，因此该耗时不能作为真实低精度推理速度结论。

### AD sample1000 layer leave-one-out sensitivity

为了观察全量化模型中哪些层更值得保留高精度，跑了 layer leave-one-out restoration：以 `all_quant` 为基准，每次将某一层的所有 Linear 恢复为 BF16，其余层保持 `weight FP8 per-channel + activation FP8 per-token dynamic(shared-input)`。样本数为 1000。

| 实验 | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | time(min) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.237 | 0.08750 | 0.223 | 0.08284 | 9.03 |
| weight-only | 0.234 | 0.8293 | 0.222 | 0.07829 | 9.01 |
| weight & act. | 0.237 | 0.08300 | 0.226 | 0.07889 | 15.08 |
| smoothquant + weight & act. | 0.233 | 0.0839 | 0.221 | 0.08179 | 18.76 |
| ranking_margin smoothquant + weight & act. | 0.231 | 0.0833 | 0.216 | 0.0788 | 17.76 |

`gain_pid_recall@32 = restore_layer_i - all_quant`，正数表示恢复该层为 BF16 有帮助。

| restore layer | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | gain_pid_recall@32 |
|---:|---:|---:|---:|---:|---:|
| 00 | 0.231 | 0.08144 | 0.218 | 0.07675 | -0.00214 |
| 01 | 0.231 | 0.08243 | 0.217 | 0.07808 | -0.00081 |
| 02 | 0.228 | 0.07759 | 0.218 | 0.07309 | -0.00580 |
| 03 | 0.231 | 0.08199 | 0.218 | 0.07777 | -0.00112 |
| 04 | 0.226 | 0.07878 | 0.214 | 0.07399 | -0.00490 |
| 05 | 0.232 | 0.08176 | 0.218 | 0.07718 | -0.00171 |
| 06 | 0.221 | 0.07794 | 0.207 | 0.07369 | -0.00520 |
| 07 | 0.234 | 0.08022 | 0.225 | 0.07588 | -0.00301 |
| 08 | 0.223 | 0.07957 | 0.210 | 0.07516 | -0.00374 |
| 09 | 0.226 | 0.07945 | 0.214 | 0.07498 | -0.00391 |
| 10 | 0.226 | 0.07712 | 0.215 | 0.07252 | -0.00637 |
| 11 | 0.231 | 0.08263 | 0.219 | 0.07789 | -0.00101 |
| 12 | 0.230 | 0.08049 | 0.218 | 0.07574 | -0.00315 |
| 13 | 0.231 | 0.08142 | 0.218 | 0.07655 | -0.00234 |
| 14 | 0.224 | 0.07791 | 0.211 | 0.07329 | -0.00560 |
| 15 | 0.221 | 0.07793 | 0.208 | 0.07337 | -0.00552 |
| 16 | 0.234 | 0.08378 | 0.220 | 0.07910 | +0.00021 |
| 17 | 0.228 | 0.07956 | 0.218 | 0.07543 | -0.00346 |
| 18 | 0.220 | 0.07947 | 0.209 | 0.07549 | -0.00340 |
| 19 | 0.229 | 0.07919 | 0.218 | 0.07457 | -0.00433 |
| 20 | 0.238 | 0.08406 | 0.226 | 0.07944 | +0.00055 |
| 21 | 0.224 | 0.07979 | 0.213 | 0.07521 | -0.00368 |
| 22 | 0.228 | 0.08051 | 0.216 | 0.07607 | -0.00282 |
| 23 | 0.225 | 0.07959 | 0.214 | 0.07550 | -0.00339 |
| 24 | 0.234 | 0.08375 | 0.224 | 0.07967 | +0.00077 |
| 25 | 0.233 | 0.08377 | 0.220 | 0.07897 | +0.00008 |
| 26 | 0.240 | 0.08579 | 0.226 | 0.08104 | +0.00215 |
| 27 | 0.227 | 0.07883 | 0.218 | 0.07484 | -0.00405 |

当前最明显有正收益的是 layer 26，`pid_recall@32` 从 all_quant 的 0.07889 恢复到 0.08104，约追回一半 baseline gap；layer 24/20/16/25 也有轻微正收益。多数层恢复后反而低于 all_quant，说明 full quant 的误差存在层间交互，简单恢复任意单层不一定带来收益。下一步可以优先围绕 layer 20/24/26 做 module-level sensitivity，拆分 attn 与 MLP 内部模块。目前结论就是高层参数对精度比较敏感


## TODO
- [x] 跑第一版对齐实验HF baseline
  FP8 weight-only
  FP8 weight + activation per-token，如果加上act掉点很多，说明量化难点在act
  ——确实对激活值的量化影响很大
- [x] 做layer/module的sensitivity，也就是做层的消融
  ——已完成 layer leave-one-out 第一版，后续需要继续做 module-level sensitivity
- [ ] 量化error和recall@32的曲线，如果相关性比较弱，那就说明量化error对推荐任务的metric相关性不大，找另外的指导方式
- [ ] 从推荐指标出发，计算sensityvity score，从而metric-aware进行量化
