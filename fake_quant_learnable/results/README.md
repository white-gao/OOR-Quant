# fake_quant_learnable/results

This directory contains local experiment outputs for the OneRec learnable PTQ work. It is intentionally organized into two groups:

- experiment run directories: end-to-end generation/evaluation outputs, kept at the top level;
- `analysis/`: small diagnostic plots and JSON files used to understand quantization behavior.

## Top-Level Experiment Runs

Top-level directories such as `baseline_w8a8_*`, `smoothquant_w8a8_*`, `m2_lwt_let_*`, and `w8a8_tail1_*` are full or partial AD benchmark runs. Most contain:

```text
<run>/<model_name>/ad/test_generated.json
<run>/eval_results.json
```

Representative full-set calib1024 runs currently used in discussion:

```text
baseline_w8a8_1p7b_ad_calib1024/
smoothquant_w8a8_fold_ad_calib1024_1p7b/
smoothquant_w8a8_alpha0p4/
m2_lwt_let_sqinit_ad_calib1024_1p7b/
w8a8_tail1_ad_calib1024_1p7b_full/
baseline_w8a8_ad_calib1024_8b_full/
smoothquant_w8a8_ad_calib1024_8b_full/
m2_lwt_let_sqinit_ad_calib1024_8b_full/
```

Historical or diagnostic runs are still kept for traceability but should not be treated as main results without checking their config:

```text
*_ad1000_*
debug_*
custom_w8a8_beam_*
w8a16_prefill_w8a8_decode_*
```

`custom_w8a8_beam_*` and `w8a16_prefill_w8a8_decode_*` are not directly comparable with the current HF `generate` baseline and should only be used as debugging records.

## analysis/

### analysis/lwt_loss_curves

Historical LWT/LWT+LET per-layer loss curve plots:

```text
m1_lwt_layer_mse_initial_vs_final*.png
m2_lwt_let_layer_mse_initial_vs_final*.png
```

These plots show optimization loss trends, not final recommendation metrics.

### analysis/smoothquant_distribution

Small SmoothQuant distribution probes, mainly for last-layer `mlp.down_proj`:

```text
sq_dist_last_down_calib3*.json
sq_dist_last_down_calib3.png
```

Use these files to inspect whether SmoothQuant actually smooths activation distributions and how it changes folded weight distributions.

### analysis/smoothquant_mse_tradeoff

SmoothQuant vs baseline W8A8 block-MSE diagnostics:

```text
sq_w8a8_layer_mse_calib3.json
sq_mse_last_down_calib3_sample0.json
sq_mse_worse_layer17_*.json/png
sq_mse_worse_layer20_*.json/png
```

The layer-wise MSE check used 3 calib samples and BF16 teacher-prefix layer inputs. It is a lightweight diagnostic, not a held-out final conclusion. Current observation: SmoothQuant lowers activation ranges strongly, but folded weights become much larger; for some layers, block output MSE increases.

## archive/

`archive/` stores files moved out of the top-level results view but not permanently deleted. The current archive is:

```text
archive/redundant_20260603/
```

It contains empty debug directories, incomplete runs without evaluation outputs, custom beam / staged prefill-decode experiments that are not comparable with the current HF `generate` baseline, and old calibration-only artifacts. Remove archive folders manually only after confirming they are no longer needed.

## logs/

No top-level `logs/` directory is kept when empty. For future background runs, create `results/logs/` as needed or redirect logs into the corresponding run directory.

## Notes

- `eval_results.json` is the source for recommendation metrics.
- `test_generated.json` is the source for generated SID beams and per-sample outputs.
- Diagnostic plots/JSON under `analysis/` are for explaining mechanisms, not leaderboard numbers.
- Before reporting a result, check the run config under `<run>/<model_name>/ad/*config.json` or `learnable_quant_config` in `test_generated.json`.
