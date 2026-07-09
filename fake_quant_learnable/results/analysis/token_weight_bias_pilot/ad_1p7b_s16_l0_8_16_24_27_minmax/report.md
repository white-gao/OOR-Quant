# Token Weight Bias Pilot: CE-Gradient vs Quantization Error

## 目的

验证当前 `GPTQ weight` 中使用的 CE-gradient token weight 是否能预测真实 token-level quantization error。

当前 token 权重定义为：

```text
I_l,t = mean_c |h_l,t,c * d CE(last prompt -> first SID token) / d h_l,t,c|
```

量化误差定义为：

```text
E_l,t = ||Block_l^fp(X_l)_t - Block_l^w8a8(X_l)_t||_2^2 / (||Block_l^fp(X_l)_t||_2^2 + eps)
```

## 设置

- 模型：OneRec-1.7B
- 数据：`data/onerec_data/benchmark-data-calib1024/ad/ad_calib.parquet`
- 样本数：16
- 层：0, 8, 16, 24, 27
- 量化误差参考：minmax W8A8 fake-quant block
- 运行方式：临时脚本 `/tmp/token_weight_bias_pilot.py`，未修改主体代码

## Layer-Level 结果


| layer | tokens | Spearman | Kendall | Pearson(log) | Top-10% overlap |
| ------: | -------: | ---------: | --------: | -------------: | ----------------: |
|     0 |  11500 |   0.3051 |  0.2059 |       0.3131 |          0.1148 |
|     8 |  11500 |   0.1369 |  0.0851 |      -0.0448 |          0.2417 |
|    16 |  11500 |   0.3472 |  0.2336 |       0.0775 |          0.2365 |
|    24 |  11500 |   0.4258 |  0.2828 |       0.3971 |          0.2765 |
|    27 |  11500 |  -0.0126 | -0.0091 |       0.0186 |          0.1530 |

平均 Spearman：`0.2405`；中位数 Spearman：`0.3051`。

弱相关层，即 `abs(Spearman) < 0.15`：`[8, 27]`。

## 主要观察

1. CE-gradient token weight 和 block quantization error 的相关性不稳定。Layer 24 有中等正相关，Layer 27 几乎无相关，Layer 8 也很弱。
2. Top-10% overlap 整体偏低，说明 CE-gradient 认为最重要的一批 token 与真实量化误差最大的 token 并不高度重合。
3. Group-level 统计在部分中间层更一致，但最后一层 group correlation 为负，说明 group 聚合能降噪但并不能根治 metric bias。
4. 这个结果支持当前判断：CE-gradient 更像 task-loss saliency，而不是稳定的 quantization-error sensitivity。

## 输出文件

- `token_rows.csv`：逐 token 原始数据
- `layer_correlation_summary.csv`：逐层相关性汇总
- `group_summary.csv`：逐层逐 group 均值与 group-level 相关性
- `layer_spearman.png`：逐层 Spearman 柱状图
- `scatter_layer_*.png`：逐层散点图

## 局限

这是 pilot，不是最终结论。样本数只有 16，量化误差使用 minmax W8A8 block，而不是 QIG 或完整 GPTQ 后的误差。不过它已经足以说明：当前 CE-gradient token weight 至少不是一个稳定、强相关的 token-level quantization sensitivity proxy。

## 下一步建议

1. 扩大到 64 或 128 个 calib 样本验证趋势。
2. 加入普通 GPTQ block 作为量化误差参考，观察相关性是否变化。
3. 做 split stability：calib A/B 上的 per-token CE-gradient rank 是否稳定。
4. 用 QIG-like block error attribution 替代 CE-gradient，比较其与真实 quantization error 的相关性。
