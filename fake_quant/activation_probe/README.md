# Activation Probe

Utilities for probing OneRec-1.7B activation outliers on AD SID prediction.

Run commands from the repository root.

## Distribution Profiling

Collect per-layer/module activation statistics and optional histogram plots:

```bash
bash fake_quant/activation_probe/run_activation_profile.sh
```

Main outputs:

```text
fake_quant/activation_probe/activation_profiles/v1.0/OneRec-1.7B-ad-sample-32/
```

## Token-Channel Plots

Draw token-channel heatmaps and 3D surface plots for selected layers/nodes:

```bash
bash fake_quant/activation_probe/run_token_channel_plots.sh
```

Main outputs:

```text
fake_quant/activation_probe/activation_profiles/v1.0/token_channel_sample_0/
```

## Channel Overlap

Analyze whether top outlier channels are fixed across layers:

```bash
bash fake_quant/activation_probe/run_channel_overlap.sh
```

Main outputs:

```text
fake_quant/activation_probe/activation_profiles/v1.0/channel_overlap_sample_0/
```

Default captured nodes:

```text
attn_qkv_input
attn_o_input
ffn_gate_up_input
ffn_down_input
```
