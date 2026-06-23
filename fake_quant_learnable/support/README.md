# fake_quant_learnable/support

This package stores support code that is useful for the active W8A8/SmoothQuant
runner but is not part of the minimal root-level API.

- `smoothquant_core.py`: standalone SmoothQuant scale and weight-folding helpers
  migrated from the legacy `fake_quant` package.
- `smoothquant_runtime.py`: SmoothQuant scale collection, exact folds, and fixed
  SQ W8A8 wrapper construction.
- `runtime_utils.py`: tensor-tree detach/device helpers used by the runner and
  analysis tools.
- `inspect_smoothquant_distribution.py`: offline SmoothQuant distribution and
  MSE inspection script. Run it with:

```bash
python3 -m fake_quant_learnable.support.inspect_smoothquant_distribution
```

- `benchmark_fp8_vs_bf16_mm_simple.py`: simple CUDA GEMM-only benchmark for
  BF16 `torch.mm` versus FP8 `torch._scaled_mm`. It prepares BF16/FP8 tensors
  before timing, then compares only the matrix multiplication kernels. Example:

```bash
python3 -m fake_quant_learnable.support.benchmark_fp8_vs_bf16_mm_simple \
  --device cuda:1 --warmup 20 --iters 100
```

- `benchmark_fp8_vs_bf16_mm.py`: extended benchmark that also has an
  `fp8_dynamic_act` timing including activation quantization overhead.

- `legacy_fake_quant_notes.md`: historical notes migrated before removing the
  legacy `fake_quant/` directory.
- `archived_lwc_let/`: historical LWC/LET and standalone tail-protect ablation
  snapshot. It is kept for reference only and should not be imported by active
  experiments. The active GPTQ+tail1 path is implemented in `quant.py`,
  `modules.py`, `gptq.py`, and `run_m1_onerec_ad.py`.
