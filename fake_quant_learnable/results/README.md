# fake_quant_learnable/results

This directory stores local OneRec AD quantization experiment outputs.

The top-level layout is intentionally split by purpose:

```text
analysis/   diagnostic plots and small JSON files
runs/       completed end-to-end generation/evaluation runs
archive/    incomplete, redundant, or non-comparable historical outputs
```

Large run directories usually contain:

```text
<run>/<model_name>/ad/test_generated.json
<run>/<model_name>/ad/*.json.debug
<run>/eval_results.json
```

`eval_results.json` is the source for recommendation metrics. `test_generated.json` is the source for generated SID beams and per-sample outputs.

## runs/

### runs/main_full_calib1024/

Completed full-test runs on the calib1024 split that are useful as primary baselines:

```text
baseline_w8a8_1p7b_ad_calib1024/
baseline_w8a8_ad_calib1024_8b_full/
smoothquant_w8a8_1p7b_ad_calib1024_fixed_fold_full/
smoothquant_w8a8_fold_ad_calib1024_1p7b/
smoothquant_w8a8_alpha0p4/
smoothquant_w8a8_ad_calib1024_epochs2_20260531_142733/
smoothquant_w8a8_ad_calib1024_8b_full/
```

### runs/ablations_full_calib1024/

Completed full-test ablation runs:

```text
baseline_w8a8_skip_last_ad_calib1024_1p7b_full/
baseline_w8a8_skip_last4_ad_calib1024_1p7b_full/
w8a8_tail1_ad_calib1024_1p7b_full/
w8a8_decode_a16_ad_calib1024_1p7b_full/
```

`w8a8_decode_a16_ad_calib1024_1p7b_full/run.log` is the background log for that run.

### runs/historical_lwc_let/

Historical LWC/LET results kept for traceability. These methods are no longer part of the active `fake_quant_learnable` runner:

```text
m2_lwt_let_1p7b_ad_calib1024_sqinit_fixed_fold_full/
m2_lwt_let_ad_calib1024_epochs2_20260529_170532/
m2_lwt_let_ad_calib1024_epochs2_20260531_143027/
m2_lwt_let_sqinit_ad_calib1024_1p7b/
m2_lwt_let_sqinit_ad_calib1024_8b_full/
```

### runs/subset_ad1000/

Old AD1000 or old-data subset runs. These are useful for historical comparison only and should not be mixed directly with full-test calib1024 conclusions:

```text
baseline_w8a8_ad1000_calib_offset_1000_20260527_130552/
m1_lwt_ad1000_calib_offset_1000_20260527_114303/
m2_let_ad1000_calib_offset_1000_20260527_141945/
m2_lwt_let_ad1000_calib_offset_1000_20260527_123634/
m2_lwt_let_ad1000_calib_offset_1000_20260527_151734/
smoothquant_w8a8_1p7b_olddata_ad1000_calib128_offset1000_fixed/
```

## analysis/

### analysis/lwt_loss_curves/

Historical M1/M2 per-layer MSE plots:

```text
m1_lwt_layer_mse_initial_vs_final*.png
m2_lwt_let_layer_mse_initial_vs_final*.png
```

These plots show optimization loss trends, not final recommendation metrics.

### analysis/smoothquant_distribution/

Small SmoothQuant distribution probes for activation/weight distribution checks:

```text
sq_dist_last_down_calib3*.json
sq_dist_last_down_calib3.png
```

### analysis/smoothquant_mse_tradeoff/

SmoothQuant vs baseline W8A8 block-MSE diagnostics:

```text
sq_w8a8_layer_mse_calib3.json
sq_mse_last_down_calib3_sample0.json
sq_mse_worse_layer17_*.json/png
sq_mse_worse_layer20_*.json/png
```

These diagnostics use a small number of calibration samples and should be treated as mechanism probes, not held-out benchmark results.

### analysis/token_modality_gap/

MQuant/MBQ-inspired probes comparing text-token and SID-token activation distributions for OneRec-1.7B and OneRec-8B.

## archive/

`archive/` stores moved-out outputs that should not clutter the active results view:

```text
archive/redundant_20260603/
```

This includes incomplete debug directories, old custom beam experiments, staged prefill/decode experiments, and calibration-only artifacts. These are not directly comparable with the current HF `generate` baseline.

## Active Top-Level Runs

Top-level run directories should normally be avoided. If a directory remains at the top level, it is likely still running or was intentionally left untouched.

As of the cleanup on 2026-06-08, this directory was left in place because a process was actively writing to it:

```text
w8a8_decode_a16_ad_calib1024_8b_full/
```

After it finishes and produces `eval_results.json`, move it to:

```text
runs/ablations_full_calib1024/
```
