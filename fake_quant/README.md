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

Layer leave-one-out sensitivity on AD sample-1000:

```bash
bash fake_quant/run_layer_sensitivity.sh
```

SmoothQuant-style FP8 fake quant:

```bash
bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

Smoke test with 2 samples and small beam:

```bash
python fake_quant/run_ad_sid.py \
  --data_dir data/onerec_data/benchmark-data \
  --output_dir fake_quant/results/v1.0/results_OneRec-1.7B-hf-fake-fp8-smoke \
  --model_name OneRec-1.7B-hf-fake-fp8-smoke \
  --sample_size 2 \
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

SmoothQuant-style FP8 fake quant:

```bash
--quant_scheme fp8_smoothquant \
--smooth_scales_path fake_quant/smoothquant/scales/onerec_ad_smoothquant_absmax_sample128.pt \
--smooth_alpha 0.5 \
--act_quant per_token \
--act_quant_mode shared_input
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
  run_ad_sid.py                      # HF AD SID generation entrypoint
  run_hf_ad_full_quant.sh            # baseline / weight-only / weight+act runs
  run_layer_sensitivity.sh           # layer leave-one-out sensitivity
  smoothquant/                       # SmoothQuant calibration and runner
  activation_probe/                  # activation outlier probing tools
  weight_probe/                      # weight distribution probing tools
  results/                           # local fake-quant benchmark outputs
```
