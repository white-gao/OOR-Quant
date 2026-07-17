# OOR-Quant

OOR-Quant 是面向 OpenOneRec 的训练后 FP8 量化研究仓库。当前稳定的使用路径包括：Plain GPTQ、面向推荐 token 的 Weighted GPTQ，以及实际部署中使用的 W8A8 prefill / W8A16 decode 策略。

本文档只介绍可直接使用的 GPTQ 相关功能。

## FP8 量化策略

仓库中的 W8A8 路径使用 PyTorch 的 FP8 E4M3FN 格式（`torch.float8_e4m3fn`）：

~~~text
权重：      FP8 E4M3FN，per-output-channel scale
激活值：    FP8 E4M3FN，per-token dynamic absmax scale
Linear 输出：BF16
lm_head：  BF16
~~~

仓库包含两类实现：

| 路径 | 位置 | 用途 |
|---|---|---|
| Real FP8 | `real_quant/naive_w8a8/` | 使用 `torch._scaled_mm` 执行真实 FP8 GEMM，用于时延和部署导向实验。 |
| Fake quant | `fake_quant_learnable/` | 使用 FP8-QDQ 模拟数值量化，用于精度分析和调试，不能用于报告推理时延。 |

## 环境与数据

请从仓库根目录运行命令。Real FP8 需要 CUDA PyTorch 和支持 FP8 `torch._scaled_mm` 的 GPU。

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(hasattr(torch, '_scaled_mm'))"
~~~

下面的命令默认使用：

~~~text
模型：
/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B

数据：
data/onerec_data/benchmark-data-calib1024
~~~

数据目录中包含独立的 calibration 和 test 划分，例如 `ad_calib.parquet` 与 `ad_test.parquet`。GPTQ 只使用 calibration 数据收集 Hessian；评测使用 test 数据。

## Real FP8 入口

GPTQ 精度与时延实验使用：

~~~bash
python -m real_quant.naive_w8a8.run_hf_naive_w8a8
~~~

与本文档相关的 `--weight_quant_mode` 如下：

| 模式 | 说明 |
|---|---|
| `minmax` | Naive FP8 W8A8 基线。 |
| `gptq` | Plain GPTQ 基线。 |
| `weighted_gptq` | 手工 role 权重：text/history-SID/interest-SID/boundary。 |
| `grad_weighted_gptq` | 由 CE 梯度估计的 role 权重。 |
| `slot_weighted_gptq` | 手工 SID slot 权重：text/SID-a/SID-b/SID-c/boundary。 |
| `slot_grad_weighted_gptq` | 由 CE 梯度估计的 SID slot 权重，当前推荐使用的 Weighted GPTQ 路径。 |

Weighted GPTQ 只改变离线 Hessian 收集与权重量化过程。量化完成后，它不会引入额外的推理算子或时延。

## 快速开始：Plain GPTQ

以下命令运行 full W8A8 的 Plain GPTQ，使用 128 条 calibration 和 1000 条独立 AD 测试样本，适合快速迭代。

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --weight_quant_mode gptq \
  --gptq_calib_sample_size 128 \
  --sample_size 1000 \
  --output_dir real_quant/naive_w8a8/results/ad_1p7b_plain_gptq_w8a8_calib128_test1000 \
  --evaluate
~~~

最终实验建议使用 `--gptq_calib_sample_size 1024`，并去掉 `--sample_size 1000`，从而评测完整 held-out test 集。

## Weighted GPTQ

当前面向推荐任务的 Weighted GPTQ 使用 `slot_grad_weighted_gptq`。它区分：

~~~text
text / boundary / SID-a / SID-b / SID-c
~~~

下面的命令使用 full-SID multi-target teacher forcing。在 calibration 阶段，最多四个 ground-truth item SID 会共同参与梯度权重估计。

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --weight_quant_mode slot_grad_weighted_gptq \
  --grad_weight_loss_mode full_sid_multi_target \
  --grad_weight_max_targets 4 \
  --gptq_calib_sample_size 128 \
  --sample_size 1000 \
  --output_dir real_quant/naive_w8a8/results/ad_1p7b_slot_grad_weighted_gptq_fullsid_mt4_w8a8_calib128_test1000 \
  --evaluate
~~~

默认梯度目标是 `first_sid`，只监督第一个生成 SID token。当前推荐任务实验建议使用 `full_sid_multi_target`。

如果需要进行手工 slot 权重消融，可以使用：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --weight_quant_mode slot_weighted_gptq \
  --slot_weight_text 10 \
  --slot_weight_sid_a 5 \
  --slot_weight_sid_b 2 \
  --slot_weight_sid_c 2 \
  --slot_weight_boundary 2 \
  --gptq_calib_sample_size 128 \
  --sample_size 1000 \
  --output_dir real_quant/naive_w8a8/results/ad_1p7b_slot_weighted_gptq_w8a8_calib128_test1000 \
  --evaluate
~~~

`weighted_gptq` 与 `grad_weighted_gptq` 使用 history/interest SID 的 role 分组，主要保留用于消融实验。

## W8A16 Decode

OneRec 每次生成固定的三个 SID token。Prefill 是大规模 GEMM，适合使用 W8A8；后续两个单 token decode 是小规模 GEMV，动态激活 FP8 量化的收益有限，反而可能引入额外开销。

实际部署可使用：

~~~text
Prefill：W8A8
Decode： W8A16
~~~

在 real-quant 命令中加入 `--decode_a16_single_token`：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --weight_quant_mode slot_grad_weighted_gptq \
  --grad_weight_loss_mode full_sid_multi_target \
  --grad_weight_max_targets 4 \
  --gptq_calib_sample_size 128 \
  --sample_size 1000 \
  --decode_a16_single_token \
  --output_dir real_quant/naive_w8a8/results/ad_1p7b_slot_grad_weighted_gptq_fullsid_mt4_decode_a16_calib128_test1000 \
  --evaluate
~~~

该选项只在当前 Linear 的序列长度为 1 时绕过 activation FP8-QDQ。权重仍为 FP8，因此 decode 是 W8A16，而不是 BF16。

比较量化算法时，应统一运行策略：

~~~text
算法比较：所有方法均使用 full W8A8
部署比较：所有量化方法均使用 W8A8 prefill / W8A16 decode
~~~

不能将 full W8A8 基线和 decode W8A16 方法直接比较后，把差异全部归因于 GPTQ 算法。

## 结果与时延

一次 real-quant 运行会写入：

~~~text
<output_dir>/
  eval_results.json
  OneRec-1.7B-real-naive-w8a8-*/ad/hf_naive_w8a8_config.json
  OneRec-1.7B-real-naive-w8a8-*/ad/test_generated.json
~~~

比较实验前，请先检查 `hf_naive_w8a8_config.json`。其中记录了量化模式、calibration 数量、梯度损失形式和 decode 策略。

GPTQ 的 Hessian 收集与权重量化都在生成前完成。 `eval_results.json` 中的 `total_time` 与 `avg_time_per_sample` 记录生成/评测时间，不包含离线 GPTQ 校准时间。

公平的 BF16 时延基线示例：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m real_quant.full_precision.run_hf_baseline \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --task ad \
  --sample_size 1000 \
  --device cuda:0 \
  --output_dir real_quant/full_precision/results/ad_1p7b_bf16_test1000 \
  --evaluate
~~~

更多 real FP8 实现和时延对比细节见 `real_quant/naive_w8a8/README.md`。

## Fake Quant 调试

Fake quant 用于检查数值精度，不用于报告时延：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m fake_quant_learnable.run_m1_onerec_ad \
  --mode gptq_fp8_w8a8 \
  --model_path /home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --calib_sample_size 128 \
  --eval_sample_size 1000 \
  --device cuda:0 \
  --output_dir fake_quant_learnable/results/runs/gptq_fp8_w8a8_calib128_test1000 \
  --evaluate
~~~

目前 fake-quant 中可用的 weighted 模式为 `weighted_gptq_fp8_w8a8` 和 `grad_weighted_gptq_fp8_w8a8`。对于 SID-slot 权重和 W8A16 decode，优先使用 real-quant runner。

## 目录结构

~~~text
real_quant/
  full_precision/       BF16 HuggingFace 基线
  naive_w8a8/           real FP8 W8A8、GPTQ、weighted GPTQ、W8A16 decode

fake_quant_learnable/   FP8-QDQ 数值模拟、算法调试和 probe
benchmarks/             RecIF-Bench 加载与评测
data/onerec_data/       calibration/test parquet 与 SID-to-PID 映射
~~~

## 使用建议

- `calib=128, test=1000` 只用于快速迭代；最终结论请使用 `calib=1024` 和完整 held-out test 集。
- 为了与默认 beam-search 评测对齐，请保持 `batch_size=1`。
- `PTQ_papers/` 是本地论文目录，不属于代码运行流程。
