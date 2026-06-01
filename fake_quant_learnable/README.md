# OneRec Learnable PTQ

This package is the isolated implementation path for SID-guided learnable PTQ.
It intentionally does not reuse the old probing, ranking-margin, SmoothQuant, or
static-activation experiment code copied from `fake_quant`.

## Quantization Semantics

Baseline, M1, and M2 use the same forward fake-quant core as the original
`fake_quant` package:

```text
weight: per-output-channel absmax scale + torch.float8_e4m3fn QDQ
activation: optional per-token absmax scale + torch.float8_e4m3fn QDQ
```

M1 only changes the weight clipping threshold before QDQ. M2 adds a learnable
LET scale per input channel and applies `x / s` plus `W * s` before activation
and weight QDQ. In M2, the LWT clipping base is recomputed from the current
LET-transformed weight `W * s`, not from the original `W`. During calibration
the FP8 forward result is preserved, while a uniform STE proxy is used only for
backward gradients so the clip/scale parameters can learn.

## Modes

```text
baseline_w8a8: min-max weight FP8 + activation FP8, no training
m1_lwt:       learnable per-output-channel weight clipping + activation FP8
m2_let:       learnable LET only, with clipping fixed at min-max
m2_lwt_let:   M1 + learnable per-input-channel LET scale
```

M1 validates the learnable PTQ chain. `m2_let` ablates LWT by training only
LET while keeping the clipping multiplier fixed. `m2_lwt_let` uses the same
loss and optimizer but trains both LWT and LET parameters:

```text
weight: frozen W + learnable per-output-channel clipping
activation: fixed per-token QDQ; `shared_input` reuses one QDQ for q/k/v and one QDQ for gate/up
loss: plain Transformer block output MSE
M1: no LET
M2-LET: learnable LET with x/s and W*s, fixed clip multiplier
M2-LWT-LET: learnable LET plus learnable clipping
M2: q/k/v share one LET parameter when names match; gate/up share one LET parameter
no SID-guided gradient weighting
no Linear-output guided loss
```

The production model weights are not updated. M1 trains
`LearnableFakeQuantLinear.log_clip_multiplier`; `m2_let` trains only
`LearnableFakeQuantLinear.log_let_scale`; `m2_lwt_let` trains both parameters.
Known Qwen-style shared-input groups
(`q_proj/k_proj/v_proj` and `gate_proj/up_proj`) share a single LET parameter;
other Linear modules keep independent LET parameters.

## 计划中的 M3：SID 引导的加权 Block 重构

M3 不改变 M2 的量化模块，仍然使用 LWT、Linear scale-only LET 和 W+A fake quant。它只改变 calibration loss：用 SID 预测的 end loss 生成固定的重要性权重，再用这个权重加权 block reconstruction MSE。

M2 的普通 block 重构目标是：

$$
\mathcal{L}_{plain}^{(l)}
= \left\|
Y_l - \hat{Y}_l
\right\|_F^2 .
$$

其中 $Y_l$ 是 BF16 teacher 的第 $l$ 个 block 输出，$\hat{Y}_l$ 是量化 student 的第 $l$ 个 block 输出。

M3 先在 BF16 teacher 上计算 teacher-forced SID loss：

$$
\mathcal{L}_{sid}
= \sum_{g=1}^{G}
\operatorname{CE}
\left(
z_g,
y_g
\right) .
$$

其中 $g$ 表示 SID 生成位置，$z_g$ 是该位置的 SID logits，$y_g$ 是目标 SID token。然后反传得到第 $l$ 个 block 输出上的梯度：

$$
G_l = \frac{\partial \mathcal{L}_{sid}}{\partial Y_l} .
$$

第一版默认使用 channel-wise importance，降低小 calibration set 下的噪声：

$$
I_l = \operatorname{mean}_{B,T}
\left(
G_l^2
\right) .
$$

这里 $I_l$ 只保留 hidden channel 维度，可以在 loss 中自动广播到 batch 和 token 维度。随后做均值归一化：

$$
\bar{I}_l
= \frac{I_l}{\operatorname{mean}(I_l) + \epsilon} .
$$

随后对归一化后的 importance 做截断，避免极端梯度权重主导优化：

$$
\tilde{I}_l
= \operatorname{clip}
\left(
\bar{I}_l,
I_{min},
I_{max}
\right) .
$$

推荐初始截断范围：

$$
I_{min}=0.5,
\qquad
I_{max}=2.0 .
$$

最终 M3 的第 $l$ 个 block calibration loss 是加权 Frobenius 范数：

$$
\mathcal{L}_{M3}^{(l)}
= \left\|
\operatorname{stopgrad}
\left(
\sqrt{\tilde{I}_l}
\right)
\odot
\left(Y_l - \hat{Y}_l\right)
\right\|_F^2 .
$$

注意：Stage B 只优化 LWT 和 LET 量化参数，不优化 SID CE、Recall@K、NDCG@K、top-k overlap、原始模型权重或 importance 权重。SID loss 只在 Stage A 中用于生成固定的 reconstruction 权重。

## Files

```text
fake_quant_learnable/
  quant.py              # FP8 E4M3 QDQ forward and STE calibration wrappers
  modules.py            # baseline, learnable, and frozen Linear wrappers
  apply.py              # Linear replacement, parameter iteration, freezing
  calibrate_m1_lwt.py   # block-output MSE calibration utility
  run_m1_onerec_ad.py   # OneRec AD baseline/M1/M2 train+test runner
  tests/                # unit tests
```

## Minimal M1 Usage

```python
import copy

from fake_quant_learnable import apply_learnable_lwt, calibrate_block_mse

teacher_block = model.model.layers[layer_idx].eval()
quant_block = copy.deepcopy(teacher_block).eval()

apply_learnable_lwt(
    quant_block,
    act_quant="per_token",
    init_clip_multiplier=1.0,
)

history = calibrate_block_mse(
    teacher_block=teacher_block,
    quant_block=quant_block,
    batches=calibration_hidden_states,
    steps=200,
    lr=1e-3,
)
```

For full OneRec calibration, `calibration_hidden_states` should come from the
quantized prefix distribution of the current layer. The interface stays
block-local so later SID-guided losses can be added without coupling the core
quantizer to one benchmark runner.

## OneRec AD Runner

Baseline smoke run:

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad \
  --mode baseline_w8a8 \
  --model_path /path/to/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data \
  --output_dir fake_quant_learnable/results/baseline_w8a8_ad_smoke \
  --layers last:1 \
  --eval_sample_size 2 \
  --num_beams 2 \
  --num_return_sequences 2 \
  --overwrite \
  --evaluate
```

M2 LET smoke run:

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad \
  --mode m2_lwt_let \
  --model_path /path/to/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data \
  --output_dir fake_quant_learnable/results/m2_lwt_let_ad_smoke \
  --layers last:1 \
  --calib_sample_size 8 \
  --calib_offset 1000 \
  --eval_sample_size 2 \
  --eval_offset 0 \
  --steps 5 \
  --lr 1e-3 \
  --num_beams 2 \
  --num_return_sequences 2 \
  --overwrite \
  --evaluate
```

M1 smoke run:

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad \
  --mode m1_lwt \
  --model_path /path/to/OneRec-1.7B \
  --data_dir data/onerec_data/benchmark-data \
  --output_dir fake_quant_learnable/results/m1_lwt_ad_smoke \
  --layers last:1 \
  --calib_sample_size 8 \
  --calib_offset 1000 \
  --eval_sample_size 2 \
  --eval_offset 0 \
  --steps 5 \
  --lr 1e-3 \
  --num_beams 2 \
  --num_return_sequences 2 \
  --overwrite \
  --evaluate
```

Recommended first matched comparison uses the same `--layers`, beam settings,
and eval set for all learnable modes. For example, compare `baseline_w8a8`,
`m1_lwt`, and `m2_lwt_let` with `--layers last:8`, `--eval_sample_size 1000`,
and `--num_beams 32`. Add the old `W+A+SmoothQuant` path as a stronger baseline
when reporting LET results.

## Current AD1000 Matched Results

Matched `fake_quant_learnable` runs on May 27, 2026 used all 28 transformer
layers, `act_quant=per_token`, `act_quant_mode=per_linear`,
`calib_sample_size=128`, `calib_offset=1000`,
`eval_sample_size=1000`, `eval_offset=0`, `steps=200`, `lr=1e-3`,
`num_beams=32`, and `num_return_sequences=32`.

| method | pass@1 | pass@16 | pass@32 | recall@1 | recall@16 | recall@32 | pid_pass@1 | pid_pass@16 | pid_pass@32 | pid_recall@1 | pid_recall@16 | pid_recall@32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_w8a8 | 0.019000 | 0.164000 | 0.233000 | 0.005891 | 0.057127 | 0.081563 | 0.016000 | 0.158000 | 0.220000 | 0.005369 | 0.054230 | 0.076784 |
| m1_lwt | 0.020000 | 0.153000 | 0.216000 | 0.005901 | 0.054812 | 0.078157 | 0.018000 | 0.145000 | 0.207000 | 0.005508 | 0.051762 | 0.074029 |
| m2_lwt_let | 0.024000 | 0.157000 | 0.224000 | 0.007194 | 0.055421 | 0.082351 | 0.022000 | 0.146000 | 0.212000 | 0.006815 | 0.052030 | 0.078189 |

Summary: M1 improves top-1 slightly but hurts top-16/top-32 coverage versus
`baseline_w8a8`. M2 is consistently better than M1 and improves recall-style
metrics over `baseline_w8a8`, but its pass@16/pass@32 coverage remains lower.

Outputs are written under:

```text
<output_dir>/<model_name>/ad/test_generated.json
<output_dir>/<model_name>/ad/m1_calibration.json, for --mode m1_lwt
<output_dir>/<model_name>/ad/m2_let_calibration.json, for --mode m2_let
<output_dir>/<model_name>/ad/m2_calibration.json, for --mode m2_lwt_let
<output_dir>/<model_name>/ad/m1_lwt_learned_quant_params.pt, for --mode m1_lwt
<output_dir>/<model_name>/ad/m2_let_learned_quant_params.pt, for --mode m2_let
<output_dir>/<model_name>/ad/m2_lwt_let_learned_quant_params.pt, for --mode m2_lwt_let
<output_dir>/<model_name>/ad/baseline_w8a8_config.json, for --mode baseline_w8a8
<output_dir>/eval_results.json, when --evaluate is set
```


Load saved learnable quantization parameters without recalibration:

```bash
LOAD_QUANT_PARAMS=fake_quant_learnable/results/<run>/<model_name>/ad/m2_lwt_let_learned_quant_params.pt \
  bash fake_quant_learnable/run_learnable_quant_ad.sh
```

The loaded params file stores only learned quantization parameters, not the full
model weights. It is intended for repeated evaluation with different beam sizes,
eval subsets, or output directories.

Notes:

- `--layers last:1` is the safest first pass.
- `--calib_only` writes the mode config without generation.
- `--save_model_state` also saves the quantized model state dict, but this can be large.
- Use the same `--act_quant` and `--act_quant_mode` for baseline, M1, and M2 when comparing.
- Current runner defaults to `ACT_QUANT_MODE=shared_input`, which quantizes q/k/v and gate/up shared inputs according to the actual module dataflow.
- Learnable modes save lightweight LWT/LET parameters by default in `*_learned_quant_params.pt`; use `--no_save_quant_params` only for throwaway runs.
- Reuse saved parameters with `--load_quant_params <path>`; this skips calibration and applies the saved frozen quantized wrappers before generation.
- Use `--eval_sample_size 1000 --eval_offset 0` for ad1000 evaluation, and set `--calib_offset 1000` so calibration starts after the eval subset.


  MODE=m2_lwt_let \
  MODEL_PATH=/home/guowei/OneRec-1.7B/ \
  MODEL_NAME=OneRec-1.7B-m2-lwt-let-calib1024-normal \
  DATA_DIR=data/onerec_data/benchmark-data-calib1024 \
  CALIB_SPLIT=calib \
  CALIB_SAMPLE_SIZE=1024 \
  CALIB_OFFSET=0 \
  EVAL_SAMPLE_SIZE=full \
  EVAL_OFFSET=0 \
  LAYERS=all \
  ACT_QUANT=per_token \
  ACT_QUANT_MODE=shared_input \
  STEPS=2048 \
  LR=3e-4 \
  DEVICE=cuda:6 \
  NUM_BEAMS=32 \
  NUM_RETURN_SEQUENCES=32 \
  MAX_NEW_TOKENS=3 \
  RUN_NAME=m2_lwt_let_1p7b_ad_calib1024_normal_steps2048_lr3e-4_full \
  OUTPUT_DIR=fake_quant_learnable/results/m2_lwt_let_1p7b_ad_calib1024_normal_steps2048_lr3e-4_full \
  OVERWRITE=1 \
  EVALUATE=1 \
  SAVE_QUANT_PARAMS=1 \
  bash fake_quant_learnable/run_learnable_quant_ad.sh