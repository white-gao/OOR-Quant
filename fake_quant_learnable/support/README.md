# fake_quant_learnable/support

This package stores support code that is useful for the active W8A8/SmoothQuant
runner but is not part of the minimal root-level API.

- `smoothquant_runtime.py`: SmoothQuant scale collection, exact folds, and fixed
  SQ W8A8 wrapper construction.
- `runtime_utils.py`: tensor-tree detach/device helpers used by the runner and
  analysis tools.
- `inspect_smoothquant_distribution.py`: offline SmoothQuant distribution and
  MSE inspection script. Run it with:

```bash
python3 -m fake_quant_learnable.support.inspect_smoothquant_distribution
```

- `archived_lwc_let/`: historical LWC/LET and tail-protect snapshot. It is kept
  for reference only and should not be imported by active experiments.
