# Real Naive W8A8 FP8 Inference

This runner replaces selected `nn.Linear` modules with `RealFP8Linear`, which
uses `torch._scaled_mm` for FP8 W8A8 GEMM.

Quantization policy:

- weight: FP8 E4M3, per-output-channel absmax scale, computed once at load time;
- activation: FP8 E4M3, per-token dynamic absmax scale, computed inside forward;
- activation sharing: fixed shared-input reuse, so q/k/v share one activation quantization and gate/up share one activation quantization;
- output: BF16 by default;
- skipped modules: `lm_head` by default.

The default `--batch_size` is `1` for accuracy alignment with the full-precision
HF baseline. Pass `--batch_size auto` only for throughput experiments.


## Run And Compare

For a one-command latency comparison, run:

```bash
DEVICE=cuda:0 SAMPLE_SIZE=100 TASK=ad bash real_quant/run_naive_w8a8_latency_compare.sh
```

The script first runs a small `torch._scaled_mm` FP8 preflight on the selected
device, then runs full precision and real naive W8A8 with the same settings,
and writes a latency table to
`real_quant/naive_w8a8/results/<task>_<model>_latency_compare.md`.

Manual equivalent commands are below. First run the matching full-precision baseline:

```bash
PYTHONPATH=. python3 -m real_quant.full_precision.run_hf_baseline \
  --model_path /home/guowei/OneRec-1.7B/ \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --output_dir real_quant/full_precision/results/ad_1p7b_bf16 \
  --task ad \
  --device cuda:0 \
  --dtype bfloat16 \
  --sample_size 100 \
  --batch_size 1 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --max_new_tokens 3 \
  --overwrite
```

Then run naive W8A8 with the same task, sample size, batch, and beam settings:

```bash
PYTHONPATH=. python3 -m real_quant.naive_w8a8.run_hf_naive_w8a8 \
  --model_path /home/guowei/OneRec-1.7B/ \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --output_dir real_quant/naive_w8a8/results/ad_1p7b \
  --task ad \
  --device cuda:0 \
  --dtype bfloat16 \
  --sample_size 100 \
  --batch_size 1 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --max_new_tokens 3 \
  --overwrite
```

Finally compare the generated JSON files:

```bash
PYTHONPATH=. python3 -m real_quant.compare_latency \
  --baseline real_quant/full_precision/results/ad_1p7b_bf16/OneRec-1.7B/ad/test_generated.json \
  --candidate real_quant/naive_w8a8/results/ad_1p7b/OneRec-1.7B-real-naive-w8a8/ad/test_generated.json
```

Use `generate_time_total` as the primary compute latency and
`end_to_end_time_total` as the user-visible total. Keep `batch_size=1` for
accuracy-aligned comparisons; use `--batch_size auto` only for separate
throughput experiments.
