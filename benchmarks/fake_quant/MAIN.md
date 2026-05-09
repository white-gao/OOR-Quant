本md文档用于记录OneRec量化算法设计的全流程，方便迭代版本管理
## Fake quant
当前市面上的量化算法主要分为基于大模型的传统派，和将大模型量化算法迁移到其他模型的“邪修”派。老油条都知道把别的领域的方法拿到自己领域来做适配就能水一篇论文，那么“邪修”派其实也是这个逻辑。目前大模型量化算法迁移这一方向主要集中在CV模型，例如ViT，DiT，Diffusion Model，ARVR，Mamba等CV领域的模型，都有人进行了量化算法的定制。他们无疑都遵循了同一种思路：

**"CV模型架构和LLM架构有比较大的区别->架构的区别导致模型参数和计算激活值的数值分布和LLM模型有较大区别->数据分布区别导致LLM量化算法无法用于CV模型->针对CV模型的特殊分布来设计专门的量化算法"**

对于我们当前的任务，有以下statement：
- 选用的量化模型为OpenOneRec开源的OneRec系列模型，有四个版本1.7B/1.7B-Pro/8B/8B-Pro，目前用OneRec-1.7B来尝试进行伪量化的尝试
- benchmark选用OpenRecRec开源的benchmark，有多个task，针对当前的业务场景，主要关注item prediction，也就是集中在ad，video，product三个task
- 目前量化的实现方法为“伪量化/fake quant.”，本质上是在推理forward过程中劫持模型layer并进行替换，劫持操作可以将模型参数量化为低精度版本（weight quant），同时也可以将layer输入x量化为低精度版本（activation quant）
- 目前的量化目标当前模型离线存储数值类型为bf16，目标量化到fp8（暂定为E4M3格式），相关资料表明，Qwen3 bf16->fp8的量化推理速度能提升一倍左右，显存可减少70%左右

## 量化策略
所有的量化层劫持实现都在quant.py中，目前实现的量化层包括以下内容：
1. fp8_weight_per_channel():
    per-channel版本的模型权重量化，在模型读取截断一次性替换模型参数数值
2. fp8_activation_per_token()：
    per-token版本的激活值动态量化，也就是根据输入的激活值，当场计算scale进行量化

### weight quant
weight是经过漫长的模型训练的，因为细粒度的梯度下降和正则化，weight整体的分布会一直被往0附近压，所以权重的数值分布一般来说是非常紧凑，平滑，整体呈现正态分布。由于权重天生的偏正态分布，所以一般来说可以直接使用min-max方法来量化，不会有太明显的异常值来拉伸量化区间。

目前已经对模型权重分布进行统计，结果在./weight_probe中，所有线性层（attn qkvo，mlp gate/up/down）权重分布均类似正态分布，数值聚集在±0.1之间。各个layernorm参数分布有所不同，普遍单峰会出现大值，但是一般情况下不对其进行量化。

### activation quant
激活值会受到不同输入数据，条件注入等因素影响，通常来说会存在相比于平均值大数倍甚至数十倍的异常值，而且常常会呈现channel-wise或者token-wise的整体异常情况。
- 从channel/维度的层面来解释，模型channel维度一般代表一种语义，如果token/整条prompt命中了某一channel的语义，那一个channel就会呈现大值；又有另外一种理论是模型自己学到的保留高信息信号的方法
- 从token层面来解释，一般来说是由attention运算机制决定的，attention 权重是per-token的，有的时候score分配不出去就会堆到某一个锚点token，导致token-wise的激活值偏大（存疑），位置编码也是token-wise加的
对于activation的量化来说，由于LLM场景下输入的不确定性，如果要保精度的话，一般是采用在线动态量化策略，如果你的分布可以比较好地预测，也可以使用离线计算scale，然后线上直接量化的策略。而量化的方法有些论文更倾向于采用percentile方法来适配activation存在大异常值的场景，来保大多数数值量化精度损失

目前已经对激活值进行初步的探测，激活值探测开销比较大，所以还没有形成统一的结论。
- 32个sample来统计不同token的激活值分布，结果在fake_quant/activation_profiles/v1.0/OneRec-1.7B-ad-sample-32/plots，不同token的分布基本都保持一直，随着层数逐渐升高，分布曲线逐渐往高值扩张，同时chat_special token出现异常高单值
- 对单个样本来细致查看各层激活值，发现在channel维度上更容易出现普遍异常值现象，token维度上比较难观察到，结果在fake_quant/activation_profiles/v1.0/token_channel_sample_0；对每层异常值channel进行统计，查看异常通道是否存在重叠现象，对该样本的统计发现对于attn和ffn前的异常值channel一般都重叠，而内部attn/ffn内部的线性层（attn o_proj/mlp down_proj）就不太重叠。但是这块还需要再进行拓展实验，一个是样本数量，不同domain样本；另外一个是token数量，目前只看了最后的100多个token，还没看到开头的sink token

## baseline
- OneRec-V2：offline weight_linear_fp8 + activation_dynamic_fp8，MoE GEMM fp8，数值敏感或者收益低的模块保留fp16（文中没提到是哪里）

### AD full fake-quant baseline

当前先在 OneRec-1.7B 的 HF fake-quant 路径上跑 AD full，样本数为 27677，beam 设置为 `num_beams=32`、`num_return_sequences=32`、`max_new_tokens=3`。这里的耗时是 HF/PyTorch fake quant 路径的 wall-clock time，只用于记录实验开销，不代表真实 FP8 kernel 加速。

| 实验 | quant 配置 | pass@1 | pass@32 | recall@32 | pid_pass@32 | pid_recall@32 | time(min) | avg(s/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HF baseline | BF16，无 fake quant | 0.0193 | 0.2166 | 0.0750 | 0.2024 | 0.0695 | 120.94 | 0.2622 |
| FP8 weight-only | Linear weight FP8 per-channel，activation BF16 | 0.0199 | 0.2112 | 0.0729 | 0.1973 | 0.0674 | 120.86 | 0.2620 |
| FP8 weight+act | Linear weight FP8 per-channel，activation FP8 per-token dynamic(shared-input) | 0.0196 | 0.2040 | 0.0701 | 0.1902 | 0.0648 | 212.64 | 0.4610 |

相对 baseline，weight-only 的 `pid_recall@32` 下降约 0.0021，weight+act 的 `pid_recall@32` 下降约 0.0047。初步看 weight FP8 本身不是主要难点，activation dynamic fake quant 带来的推荐指标扰动更明显；同时 fake quant 的 activation QDQ 会显著增加 PyTorch 路径耗时，因此该耗时不能作为真实低精度推理速度结论。


## TODO
- [ ] 跑第一版对齐实验HF baseline
  FP8 weight-only
  FP8 weight + activation per-token，如果加上act掉点很多，说明量化难点在act
- [ ] 做layer/module的sensitivity，也就是做层的消融
- [ ] 量化error和recall@32的曲线，如果相关性比较弱，那就说明量化error对推荐任务的metric相关性不大，找另外的指导方式
