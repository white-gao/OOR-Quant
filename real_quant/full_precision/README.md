# HF Full-Precision Baseline

This directory contains the standalone HuggingFace full-precision baseline for
real FP8 runtime comparisons. It is intentionally separated from
`fake_quant_learnable` so BF16 latency, real naive W8A8, real GPTQ, and real
weighted GPTQ can later share one runner without mixing in fake-quant code.

## OpenOneRec Alignment

The implementation follows the public OpenOneRec benchmark structure:

- data loading uses `benchmarks/benchmark/tasks/v1_0/registry.py::get_loader`;
- metric evaluation uses `benchmark.Benchmark.evaluate_dev`;
- outputs are saved as `<output>/<model>/<task>/<split>_generated.json`;
- recommendation prompts append `<|sid_begin|>` before SID generation;
- default recommendation generation uses `num_beams=32`,
  `num_return_sequences=32`, and `max_new_tokens=3`, matching the public
  `benchmarks/eval_script.sh` overrides for ad/product/video-style tasks.

The intentional difference is the backend: this runner uses
`transformers.AutoModelForCausalLM.generate` instead of vLLM/Ray. That keeps the
baseline comparable to upcoming real FP8 PyTorch module replacements.

## Usage

AD full precision, one GPU:

```bash
PYTHONPATH=. python3 -m real_quant.full_precision.run_hf_baseline \
  --model_path /home/guowei/OneRec-1.7B/ \
  --data_dir data/onerec_data/benchmark-data-calib1024 \
  --output_dir real_quant/full_precision/results/ad_1p7b_bf16 \
  --task ad \
  --device cuda:0 \
  --dtype bfloat16 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --max_new_tokens 3 \
  --overwrite \
  --evaluate
```

`--batch_size` defaults to `1`. This is the safest setting for accuracy
alignment because it avoids any batch-coupled effects from padding, batched
kernel choices, or future activation quantization scale implementations.

For throughput experiments, pass `--batch_size auto` explicitly. The runner then
checks the selected CUDA device total memory, infers the model size from the
model path, and chooses a conservative throughput-oriented batch size. On the
current 140GB-class cards, the auto values are:

- OneRec-1.7B: `ad/product/label_cond=8`, `video/interactive=4`;
- OneRec-8B: `ad/product/label_cond=4`, `video/interactive=2`.

Override it explicitly, for example `--batch_size 2`, if the card is already
heavily occupied or if a task OOMs. With beam search, the effective active
sequences are roughly `batch_size * num_beams`, so video and 8B runs need lower
batches than ad/product.

## Latency Fields

The generated JSON keeps the OpenOneRec-compatible fields and adds:

- top-level `latency`: aggregate tokenize/generate/decode/end-to-end stats;
- per-sample `latency`: token counts and timing;
- per-sample `input_tokens`, `output_tokens`, and `times`, compatible with the
  benchmark MFU-style schema.

For `batch_size > 1`, batch wall time is distributed across samples, so the
reported per-sample values are batch-amortized latency. With `batch_size=1`,
they are single-request latency.

For BF16 vs real FP8 comparisons, use `generate_time_*` as the primary compute
latency and keep `end_to_end_time_*` as the user-visible total.

## TODO

- Add a unified real-quant entrypoint, tentatively `real_quant.run_hf_quant`,
  with `--mode full_precision`, `--mode naive_w8a8`, and future modes such as
  `gptq_w8a8`, `weighted_gptq_w8a8`, and `weighted_gptq_w8a8_tail1`. Keep the
  method implementations in parallel modules instead of folding future methods
  into `naive_w8a8`, while using the unified entrypoint for comparable latency
  and evaluation commands.
