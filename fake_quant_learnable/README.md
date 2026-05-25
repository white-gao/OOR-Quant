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
m2_lwt_let:   M1 + learnable per-input-channel LET scale
```

M1 validates the learnable PTQ chain; M2 uses the same loss and optimizer but
adds LET parameters:

```text
weight: frozen W + learnable per-output-channel clipping
activation: fixed per-token QDQ
loss: plain Transformer block output MSE
M1: no LET
M2: learnable LET with x/s and W*s
M2: q/k/v share one LET parameter when names match; gate/up share one LET parameter
no SID-guided gradient weighting
no Linear-output guided loss
```

The production model weights are not updated. M1 trains
`LearnableFakeQuantLinear.log_clip_multiplier`; M2 additionally trains
`LearnableFakeQuantLinear.log_let_scale`. Known Qwen-style shared-input groups
(`q_proj/k_proj/v_proj` and `gate_proj/up_proj`) share a single LET parameter;
other Linear modules keep independent LET parameters.

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
  --eval_batch_size 1 \
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
  --calib_batch_size 1 \
  --eval_batch_size 1 \
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
  --calib_batch_size 1 \
  --eval_batch_size 1 \
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

Outputs are written under:

```text
<output_dir>/<model_name>/ad/test_generated.json
<output_dir>/<model_name>/ad/m1_calibration.json, for --mode m1_lwt
<output_dir>/<model_name>/ad/m2_calibration.json, for --mode m2_lwt_let
<output_dir>/<model_name>/ad/baseline_w8a8_config.json, for --mode baseline_w8a8
<output_dir>/eval_results.json, when --evaluate is set
```

Notes:

- `--layers last:1` is the safest first pass.
- `--calib_only` writes the mode config without generation.
- `--save_model_state` also saves the quantized model state dict, but this can be large.
- Use the same `--act_quant` for baseline, M1, and M2 when comparing.
- Use `--eval_sample_size 1000 --eval_offset 0` for ad1000 evaluation, and set `--calib_offset 1000` so calibration starts after the eval subset.
