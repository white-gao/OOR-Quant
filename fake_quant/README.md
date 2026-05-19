# OneRec HF Fake Quant

This directory runs AD-domain SID prediction through HuggingFace/PyTorch rather than vLLM, with FP8 fake quantization for Linear layers. It also contains activation/weight probing and SmoothQuant-style baselines.

Default model paths can be overridden with `MODEL_PATH`. On this machine the common path is:

```bash
/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B
```

## Common Runs

HF baseline, FP8 weight-only, and FP8 weight+activation fake quant:

```bash
bash fake_quant/run_hf_ad_full_quant.sh
```

SmoothQuant-style FP8 fake quant:

```bash
bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

Generation and activation calibration support batching. The shell runners default to
`BATCH_SIZE=1` and `CALIB_BATCH_SIZE=1` for conservative memory use; increase them
when GPU memory allows. Effective generation memory scales with
`BATCH_SIZE * NUM_BEAMS`.

```bash
BATCH_SIZE=4 CALIB_BATCH_SIZE=8 bash fake_quant/run_hf_ad_full_quant.sh
```

SmoothQuant only on lower layers, with higher layers falling back to plain FP8 fake quant:

```bash
SMOOTH_LAYER_CUTOFF=20 bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

SmoothQuant only on higher layers, with lower layers falling back to plain FP8 fake quant:

```bash
SMOOTH_LAYER_MIN=20 bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

Smoke test with 2 samples and small beam:

```bash
python fake_quant/run_ad_sid.py \
  --data_dir data/onerec_data/benchmark-data \
  --output_dir fake_quant/results/v1.0/results_OneRec-1.7B-hf-fake-fp8-smoke \
  --model_name OneRec-1.7B-hf-fake-fp8-smoke \
  --sample_size 2 \
  --batch_size 2 \
  --num_beams 2 \
  --num_return_sequences 2 \
  --overwrite \
  --evaluate
```

## Quantization Options

Weight-only FP8 fake quant:

```bash
--quant_scheme fp8_weight_channel --act_quant none
```

Weight + dynamic activation FP8 fake quant:

```bash
--quant_scheme fp8_weight_channel --act_quant per_token --act_quant_mode shared_input
```

Weight + static activation FP8 fake quant:

```bash
python fake_quant/smoothquant/collect_smooth_scales.py \
  --sample_size 128 \
  --batch_size 8 \
  --sample_offset 1000 \
  --output_path fake_quant/static_activation/scales/onerec_ad_static_tensor_absmax_sample128_offset1000.pt

python fake_quant/run_ad_sid.py \
  --batch_size 4 \
  --quant_scheme fp8_weight_channel \
  --act_quant static_tensor \
  --act_quant_mode shared_input \
  --static_act_scales_path fake_quant/static_activation/scales/onerec_ad_static_tensor_absmax_sample128_offset1000.pt
```

The convenience script can run this branch with:

```bash
RUN_WEIGHT_ONLY=false RUN_WEIGHT_ACT_STATIC=true bash fake_quant/run_hf_ad_full_quant.sh
```

SmoothQuant-style FP8 fake quant:

```bash
--quant_scheme fp8_smoothquant \
--smooth_scales_path fake_quant/smoothquant/scales/onerec_ad_smoothquant_absmax_sample128_offset1000.pt \
--smooth_alpha 0.5 \
--act_quant per_token \
--act_quant_mode shared_input \
--smooth_layer_min 20 \
--smooth_layer_cutoff 20
```

Optional layer selection/restoration:

```bash
--target_regex 'model\.layers\.\d+\.mlp\.(gate_proj|up_proj)$'
--skip_regex '^model\.layers\.0\.'
```

## Directory Layout

```text
fake_quant/
  quant.py / modules.py / apply.py   # FP8 fake-quant implementation
  static_activation/                  # static activation scale utilities
  run_ad_sid.py                      # HF AD SID generation entrypoint
  run_hf_ad_full_quant.sh            # baseline / weight-only / weight+act runs
  smoothquant/                       # SmoothQuant calibration and runner
  ranking_margin/                    # ranking-margin SmoothQuant variant
  probes/                            # activation / weight probing tools
    activation_probe/
    weight_probe/
  tests/                             # unit tests for fake-quant utilities
  results/                           # local fake-quant benchmark outputs
```
