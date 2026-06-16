# OneRec-8B Text/SID Activation Distribution Probe

This directory contains MQuant Fig.1(b)-style activation distribution probes for OneRec-8B.

## Command

```bash
python3 -m fake_quant.probes.activation_probe.plot_token_modality_distribution \
  --model_path /home/guowei/OneRec-8B/ \
  --device cuda:0 \
  --sample_size 5 \
  --sample_offset 0 \
  --max_history_sid_items 0 \
  --layers 0,12,24,35 \
  --nodes attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input \
  --sid_boundary_as_sid \
  --capture_max_values_per_modality 5000 \
  --plot_max_values 30000 \
  --skip_abs_tail \
  --output_dir fake_quant_learnable/results/analysis/token_modality_gap/mquant_style_onerec8b_sample5_fullhist_layers0_12_24_35_sid_boundary_cuda0
```

## Setup

- Model: `/home/guowei/OneRec-8B/`
- Dataset: `data/onerec_data/benchmark-data-calib1024/ad`
- Samples: first 5 AD calib1024 samples
- History compression: none
- SID boundary handling: `<|sid_begin|>` and `<|sid_end|>` are counted as SID tokens
- Layers: 0, 12, 24, 35
- Probe nodes: `attn_qkv_input`, `attn_o_input`, `ffn_gate_up_input`, `ffn_down_input`

## Token Counts

| sample | prompt tokens | text tokens | SID tokens | history SID items |
|---:|---:|---:|---:|---:|
| 0 | 676 | 68 | 601 | 120 |
| 1 | 581 | 68 | 506 | 101 |
| 2 | 682 | 74 | 601 | 120 |
| 3 | 1017 | 74 | 936 | 187 |
| 4 | 944 | 71 | 866 | 173 |

## Main Ratios

Values below are `SID / text`.

| layer | node | mean_abs | p99_abs | p999_abs | absmax |
|---:|---|---:|---:|---:|---:|
| 0 | attn_qkv_input | 1.093 | 1.012 | 0.628 | 0.252 |
| 0 | attn_o_input | 1.056 | 1.007 | 0.993 | 0.906 |
| 0 | ffn_gate_up_input | 1.299 | 1.248 | 1.468 | 2.562 |
| 0 | ffn_down_input | 3.284 | 3.744 | 3.088 | 0.393 |
| 12 | attn_qkv_input | 0.836 | 1.053 | 0.664 | 0.679 |
| 12 | attn_o_input | 0.757 | 0.754 | 0.738 | 0.718 |
| 12 | ffn_gate_up_input | 0.833 | 0.918 | 0.840 | 0.767 |
| 12 | ffn_down_input | 0.672 | 0.765 | 0.744 | 0.605 |
| 24 | attn_qkv_input | 0.744 | 0.969 | 0.791 | 0.371 |
| 24 | attn_o_input | 1.246 | 1.642 | 1.817 | 2.686 |
| 24 | ffn_gate_up_input | 0.786 | 0.902 | 0.963 | 0.704 |
| 24 | ffn_down_input | 0.728 | 0.700 | 0.828 | 1.744 |
| 35 | attn_qkv_input | 0.676 | 0.876 | 0.692 | 0.890 |
| 35 | attn_o_input | 0.674 | 0.731 | 0.706 | 1.157 |
| 35 | ffn_gate_up_input | 0.661 | 0.785 | 0.666 | 0.389 |
| 35 | ffn_down_input | 1.127 | 1.040 | 1.255 | 2.000 |

## Observation

The 8B probe does not show a stable global MLLM-style modality dynamic-range gap where SID tokens are always much larger than text tokens. Instead, the gap is local and module-dependent. The strongest SID-over-text regions are layer 0 `ffn_down_input` and layer 24 `attn_o_input`; many middle/high-layer block inputs have SID ratios below 1.

This is consistent with the 1.7B observation: SID tokens tend to have different concentration/shape, but not a uniformly larger activation dynamic range across all layers and nodes.
