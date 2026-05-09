# OneRec HF Fake Quant

This directory runs AD-domain SID prediction through HuggingFace/PyTorch rather
than vLLM, with simple FP8 QDQ fake quantization for Linear layers.

Default base model:

```bash
/home/guowei/OneRec-1.7B
```

Smoke test with 2 samples and small beam:

```bash
python fake_quant/run_ad_sid.py \
  --output_dir results/v1.0/results_OneRec-1.7B-hf-fake-fp8-smoke \
  --model_name OneRec-1.7B-hf-fake-fp8-smoke \
  --sample_size 2 \
  --num_beams 2 \
  --num_return_sequences 2 \
  --overwrite \
  --evaluate
```

AD sample-1000 run aligned with the current benchmark beam setting:

```bash
python fake_quant/run_ad_sid.py \
  --output_dir results/v1.0/results_OneRec-1.7B-hf-fake-fp8-weight-channel-ad-sample-1000 \
  --model_name OneRec-1.7B-hf-fake-fp8-weight-channel \
  --sample_size 1000 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --overwrite \
  --evaluate
```

HF baseline for comparison:

```bash
python fake_quant/run_ad_sid.py \
  --quant_scheme none \
  --output_dir results/v1.0/results_OneRec-1.7B-hf-baseline-ad-sample-1000 \
  --model_name OneRec-1.7B-hf-baseline \
  --sample_size 1000 \
  --num_beams 32 \
  --num_return_sequences 32 \
  --overwrite \
  --evaluate
```

Full AD runs for baseline, weight-only, and weight+activation fake quant:

```bash
bash fake_quant/run_hf_ad_full_quant.sh
```

The full script writes separate result directories under `fake_quant/results/v1.0/`.

Optional activation QDQ:

```bash
--act_quant per_token
```

By default, activation QDQ is applied inside each replaced Linear wrapper. To
reuse the same QDQ result for shared inputs such as `q/k/v` and `gate/up`, use:

```bash
--act_quant per_token --act_quant_mode shared_input
```

Optional layer selection, for example MLP gate/up only:

```bash
--target_regex 'model\\.layers\\.\\d+\\.mlp\\.(gate_proj|up_proj)$'
```

## Directory Layout

```text
fake_quant/
  quant.py / modules.py / apply.py   # FP8 fake-quant implementation
  run_ad_sid.py                      # HF AD SID generation entrypoint
  run_hf_ad_compare.sh               # HF baseline vs fake-quant comparison
  activation_probe/                  # activation outlier probing tools
  activation_profiles/               # local activation probing outputs
  results/                           # local fake-quant benchmark outputs
```

Activation probing commands are under `activation_probe/`:

```bash
bash fake_quant/activation_probe/run_activation_profile.sh
bash fake_quant/activation_probe/run_token_channel_plots.sh
bash fake_quant/activation_probe/run_channel_overlap.sh
```
