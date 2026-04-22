# AD 1000 Beam Rank Analysis

## Purpose

本分析用于解释 FP8 e4m3 weight QDQ 后，推荐排序具体发生了什么变化。它不是新的量化算法，而是把最终 metric 变化拆解成：

```text
top-1 是否变化
top-32 candidate set 是否稳定
target SID/PID rank 是否移动
target 是否从 top-32 掉出或进入 top-32
beam logprob margin 是否变化
```

分析对象是 AD 1000 条子集：

```text
../data/onerec_data/benchmark_data/ad/ad_test_sample_1000.parquet
```

输入文件来自逐样本 generation 结果，而不是聚合指标：

```text
results/v1.0/results_OneRec-1.7B_ad_sample_1000/OneRec-1.7B/ad/test_generated.json
results/v1.0/results_OneRec-1.7B-fp8e4m3-mlp-all_ad_sample_1000/OneRec-1.7B-fp8e4m3-mlp-all/ad/test_generated.json
results/v1.0/results_OneRec-1.7B-fp8e4m3-block-linears_ad_sample_1000/OneRec-1.7B-fp8e4m3-block-linears/ad/test_generated.json
```

生成脚本：

```bash
python scripts/analyze_beam_rank_shift.py
```

主要输出：

```text
results/v1.0/beam_rank_ad_sample_1000_samples.csv
results/v1.0/beam_rank_ad_sample_1000_summary.md
results/v1.0/beam_rank_ad_sample_1000_examples.md
```

## Metric Check

1000 子集上的聚合指标如下：

| setting | pass@1 | pass@32 | recall@32 | pid_pass@1 | pid_pass@32 | pid_recall@32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.023 | 0.238 | 0.087106 | 0.021 | 0.223 | 0.082444 |
| mlp-all | 0.023 | 0.235 | 0.083516 | 0.020 | 0.220 | 0.078597 |
| block-linears | 0.022 | 0.235 | 0.084288 | 0.019 | 0.223 | 0.079494 |

在该子集上，`mlp-all` 和 `block-linears` 的 top-32 指标变化都很小。后续排序分析主要解释这些小幅变化来自哪里。

## Candidate Set Stability

QDQ 后 top-1 变化比例较高，但 top-32 candidate set 仍保留了较多 baseline 候选。

| setting | SID top1 changed | SID top32 overlap avg | SID Jaccard avg | PID top1 changed | PID top32 overlap avg | PID Jaccard avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp-all | 37.0% | 25.23 / 32 | 0.654 | 37.0% | 25.09 / 32 | 0.655 |
| block-linears | 38.7% | 24.57 / 32 | 0.627 | 38.7% | 24.45 / 32 | 0.628 |

结论：

```text
FP8 QDQ 会明显扰动 top-1 和 beam 内部排序；
但 top-32 候选集合没有完全重排，平均仍有约 24-25 个候选与 baseline 相同。
block-linears 的扰动略强于 mlp-all，符合其量化参数更多的直觉。
```

## Lost and Gained

定义：

```text
stable_hit: baseline 和 QDQ 都在 top-32 命中 target
lost      : baseline top-32 命中，QDQ top-32 未命中
gained    : baseline top-32 未命中，QDQ top-32 命中
both_miss : baseline 和 QDQ 都未命中
```

SID 维度统计：

| setting | stable_hit | lost | gained | both_miss |
| --- | ---: | ---: | ---: | ---: |
| mlp-all | 219 | 19 | 16 | 746 |
| block-linears | 214 | 24 | 21 | 741 |

PID 维度统计：

| setting | stable_hit | lost | gained | both_miss |
| --- | ---: | ---: | ---: | ---: |
| mlp-all | 203 | 20 | 17 | 760 |
| block-linears | 200 | 23 | 23 | 754 |

结论：

```text
lost 和 gained 数量接近，说明 QDQ 不是单向地破坏所有样本；
更像是在 top-32 边界附近引入排序扰动；
最终指标下降来自 lost 略多于 gained，以及 recall 命中数的轻微变化。
```

## Target Rank Movement

`stable_hit` 的 target rank 通常较靠前：

| setting | stable_hit baseline target rank median | stable_hit QDQ target rank median |
| --- | ---: | ---: |
| mlp-all | 8 | 8 |
| block-linears | 8 | 9 |

`lost` 样本中，baseline target rank 明显偏后：

| setting | lost count | baseline target rank median | baseline lost ranks |
| --- | ---: | ---: | --- |
| mlp-all | 19 | 22 | 7, 10, 13, 15, 17, 17, 20, 21, 21, 22, 25, 27, 28, 29, 29, 30, 32, 32, 32 |
| block-linears | 24 | 24.5 | 7, 9, 10, 13, 14, 17, 21, 21, 22, 23, 23, 24, 25, 26, 27, 29, 29, 29, 30, 32, 32, 32, 32, 32 |

`gained` 样本中，QDQ target rank 也明显偏后：

| setting | gained count | QDQ target rank median | QDQ gained ranks |
| --- | ---: | ---: | --- |
| mlp-all | 16 | 20.5 | 8, 14, 15, 15, 16, 16, 17, 20, 21, 21, 22, 24, 26, 27, 27, 32 |
| block-linears | 21 | 22 | 9, 10, 14, 16, 16, 17, 17, 18, 20, 21, 22, 22, 23, 23, 23, 23, 26, 26, 26, 29, 31 |

核心结论：

```text
lost:   原本排在 top-32 后半段的 target 更容易被挤出 top-32；
gained: 新进入 top-32 的 target 也大多排在后半段；
stable_hit: target rank 明显更靠前，通常更稳定。
```

因此当前 QDQ 的排序影响更像是：

```text
top-32 边界候选交换
```

而不是：

```text
把强命中样本大面积破坏
或把 missed target 大量推到前排
```

## SID vs PID

SID 和 PID 的变化在该子集上高度一致：

```text
mlp-all SID Jaccard 0.654, PID Jaccard 0.655
block-linears SID Jaccard 0.627, PID Jaccard 0.628
```

lost/gained 数量也接近：

```text
mlp-all SID lost/gained = 19/16, PID lost/gained = 20/17
block-linears SID lost/gained = 24/21, PID lost/gained = 23/23
```

结论：

```text
当前 AD 1000 子集里，SID 排序变化基本能反映 PID 推荐变化；
暂时没有看到“SID 变化很多但 PID 很稳定”的强现象。
```

## Margin Observation

当前脚本使用 top1-top2 cumulative logprob margin：

```text
margin = logprob(top1) - logprob(top2)
```

平均 margin 变化：

| setting | baseline margin avg | QDQ margin avg | delta |
| --- | ---: | ---: | ---: |
| mlp-all | 0.304 | 0.335 | +0.030 |
| block-linears | 0.304 | 0.332 | +0.028 |

`lost` 样本的 baseline top1-top2 margin 没有明显更小：

```text
mlp-all lost baseline margin avg = 0.338
block-linears lost baseline margin avg = 0.335
```

结论：

```text
top1-top2 margin 不是解释 lost 的强变量；
更相关的可能是 target 附近的 margin，尤其是 target 与 rank32/rank33 边界候选的差距。
当前 test_generated.json 没有保存完整 beam tree 或 token-level logprob，因此这部分需要后续扩展。
```

## Overall Conclusion

基于 AD 1000 子集，当前可以认为：

```text
FP8 e4m3 weight QDQ 对推荐排序的影响主要表现为低幅度 beam 边界扰动。
```

更具体地说：

```text
1. top1 推荐有约 37%-39% 发生变化，说明排序前列受量化扰动明显。
2. top32 candidate set 仍保留约 24-25/32 个 baseline 候选，说明候选集合没有整体崩坏。
3. lost 和 gained 数量接近，说明扰动有一定双向性，不是单向退化。
4. lost target 通常原本就在 top32 后半段；gained target 进入后也多位于后半段。
5. stable_hit 的 target rank 明显更靠前，因此更不容易受量化影响。
6. block-linears 扰动强于 mlp-all，但两者在 1000 子集上的 pass@32 下降都很小。
```

这为后续量化研究提供了一个明确方向：

```text
RecLLM 量化不应只看 weight MSE 或最终 pass@k；
更应该关注 SID/PID top-k candidate set stability、target rank shift 和 top-k 边界样本。
```

## Next Step

建议把同样的 1000 子集 beam/rank 分析扩展到：

```text
gate-only
up-only
gate-up
mlp-down
attn-all
```

重点验证 `gate-up` 的异常下降是否也表现为：

```text
lost 更多
target 更容易从 top-32 后半段掉出
top32 overlap 更低
```
