# Ranking-Margin SmoothQuant

这个目录实现一个面向推荐指标的 SmoothQuant 变体。目标不是对 `alpha` 做普通网格搜索，而是把 Recall@K 相关的排序边界信息转成 channel importance，再用于修正 SmoothQuant 的 per-channel scale。

## Motivation

普通 SmoothQuant 的 scale 只看 activation outlier 和 weight range：

$$
s_c = \frac{a_c^\alpha}{w_c^{1-\alpha}}
$$

其中 $a_c$ 是 calibration activation absmax，$w_c$ 是对应 Linear weight 输入 channel 的 absmax。这个公式优化的是低精度矩阵乘的数值范围分配，但没有显式关注推荐任务中的 top-k 排序边界。

推荐 SID prediction 的核心风险是：量化扰动让正样本 SID 从 top-k 边界内掉出去。因此这里用 margin surrogate 估计“哪些 channel 对排序边界更敏感”。

## Objective

对 calibration 样本 $i$，设正 SID token 的 logit 为 $z_i^+$，第 $K$ 个 hard negative token 的 logit 为 $z_i^-$，margin 为：

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

## Implementation

文件结构：

```text
fake_quant/ranking_margin/
  core.py                         # scale 公式、importance 读写、group-wise scale 计算
  collect_importance.py            # 用 calibration 样本反传 ranking loss，采集 channel importance
  run_ranking_margin_smoothquant_ad.sh
  METHOD.md
```

流程：

1. 用 `fake_quant/smoothquant/collect_smooth_scales.py` 采集 activation absmax。
2. 用 `fake_quant/ranking_margin/collect_importance.py` 采集 ranking-margin channel importance。
3. 在 `fake_quant/run_ad_sid.py --quant_scheme fp8_smoothquant` 中传入 `--smooth_rank_importance_path`，评测时计算 ranking-aware scale。

默认仍然只量化 Linear 的 weight FP8 per-channel，并对 activation 做 FP8 per-token fake quant；attention 内部 BMM 不做低精度模拟。

## Commands

一键运行：

```bash
cd /zssd/home/yhhuang/Projects/OOR-Quant
CUDA_VISIBLE_DEVICES=7 bash fake_quant/ranking_margin/run_ranking_margin_smoothquant_ad.sh
```

默认配置会保持当前测试集为 `sample_size=1000`，并从 test 全集的第 1000 条之后取 calibration：

```text
eval:  rows [0, 1000)
calib: rows [1000, 1000 + CALIB_SAMPLE_SIZE)
```

对比 `128/256` 两档非重叠 calibration 时，分别调用两个原有脚本即可。

SmoothQuant：

```bash
CUDA_VISIBLE_DEVICES=7 CALIB_SAMPLE_SIZE=128 bash fake_quant/smoothquant/run_smoothquant_ad.sh
CUDA_VISIBLE_DEVICES=7 CALIB_SAMPLE_SIZE=256 bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

Ranking-Margin SmoothQuant：

```bash
CUDA_VISIBLE_DEVICES=7 CALIB_SAMPLE_SIZE=128 IMPORTANCE_SAMPLE_SIZE=128 \
bash fake_quant/ranking_margin/run_ranking_margin_smoothquant_ad.sh

CUDA_VISIBLE_DEVICES=7 CALIB_SAMPLE_SIZE=256 IMPORTANCE_SAMPLE_SIZE=256 \
bash fake_quant/ranking_margin/run_ranking_margin_smoothquant_ad.sh
```

只采集 ranking importance：

```bash
python fake_quant/ranking_margin/collect_importance.py \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data \
  --sample_size 128 \
  --sample_offset 1000 \
  --output_path fake_quant/ranking_margin/importances/onerec_ad_rank_importance_sample128_offset1000.pt
```

用已有 absmax 和 importance 评测：

```bash
python fake_quant/run_ad_sid.py \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data \
  --sample_size 1000 \
  --quant_scheme fp8_smoothquant \
  --act_quant per_token \
  --act_quant_mode shared_input \
  --smooth_scales_path fake_quant/smoothquant/scales/onerec_ad_smoothquant_absmax_sample128_offset1000.pt \
  --smooth_rank_importance_path fake_quant/ranking_margin/importances/onerec_ad_rank_importance_sample128_offset1000.pt \
  --smooth_importance_beta 0.25 \
  --output_dir fake_quant/results/v1.0/results_OneRec-1.7B-hf-fake-fp8-ranking-margin-smoothquant-ad-1000 \
  --model_name OneRec-1.7B-hf-fake-fp8-ranking-margin-smoothquant \
  --evaluate \
  --overwrite
```

## Limitations

当前 ranking loss 是 token-level surrogate，不是完整 beam-search SID sequence 的 exact margin。它的作用是用可反传、低成本的方式近似推荐 top-k 边界敏感性。

importance 采集需要反向传播，比普通 SmoothQuant calibration 慢；建议先用 `sample_size=128` 或 `256` 做方法验证，再扩大 calibration 样本。
