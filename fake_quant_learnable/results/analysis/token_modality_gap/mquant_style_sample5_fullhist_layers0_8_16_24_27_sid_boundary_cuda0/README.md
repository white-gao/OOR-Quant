# Token Modality Activation Gap Probe: Full History, SID Boundaries Included

This run extends the MQuant Figure 1(b)-style probe to full OneRec AD prompts. History SID interactions are not compressed. `<|sid_begin|>` and `<|sid_end|>` are counted as SID tokens together with `<s_a_*>`, `<s_b_*>`, and `<s_c_*>`.

Run command:

> Legacy reproduction note: this command referenced the removed `fake_quant/` probe package. Keep this README as historical provenance; rerunning requires restoring or rewriting the old probe.

```bash
python3 -m fake_quant.probes.activation_probe.plot_token_modality_distribution \
  --device cuda:0 \
  --sample_size 5 \
  --sample_offset 0 \
  --max_history_sid_items 0 \
  --layers 0,8,16,24,27 \
  --nodes attn_qkv_input,attn_o_input,ffn_gate_up_input,ffn_down_input \
  --sid_boundary_as_sid \
  --capture_max_values_per_modality 5000 \
  --plot_max_values 30000 \
  --skip_abs_tail \
  --output_dir fake_quant_learnable/results/analysis/token_modality_gap/mquant_style_sample5_fullhist_layers0_8_16_24_27_sid_boundary_cuda0
```

Probe design:

- low layer: 0
- middle layers: 8, 16
- high layers: 24, 27
- positions: attention shared input, attention output projection input, FFN gate/up shared input, FFN down input
- plots: signed activation distribution only, to match the spirit of MQuant Fig.1(b)

Token counts:

| sample | prompt tokens | text tokens | SID tokens | ignored tokens | SID items kept/original |
|---|---:|---:|---:|---:|---:|
| 0 | 676 | 68 | 601 | 7 | 120/120 |
| 1 | 581 | 68 | 506 | 7 | 101/101 |
| 2 | 682 | 74 | 601 | 7 | 120/120 |
| 3 | 1017 | 74 | 936 | 7 | 187/187 |
| 4 | 944 | 71 | 866 | 7 | 173/173 |

Largest SID/text activation magnitude ratios by p99:

| layer | node | mean_abs | p99_abs | p999_abs | absmax |
|---:|---|---:|---:|---:|---:|
| 0 | ffn_down_input | 1.503 | 3.353 | 1.845 | 0.719 |
| 16 | ffn_down_input | 1.269 | 1.154 | 1.712 | 1.842 |
| 24 | attn_qkv_input | 0.970 | 1.110 | 0.913 | 1.042 |
| 0 | ffn_gate_up_input | 1.339 | 1.029 | 1.488 | 2.833 |
| 16 | ffn_gate_up_input | 1.111 | 1.013 | 1.030 | 0.724 |

Average SID/text p99 ratio by layer:

| layer | avg p99 ratio |
|---:|---:|
| 0 | 1.553 |
| 8 | 0.883 |
| 16 | 1.012 |
| 24 | 0.821 |
| 27 | 0.813 |

Initial observation: full-history SID tokens do not exhibit a globally wider activation distribution like MQuant's visual-token case. The clearest SID-over-text signal appears in early-layer `ffn_down_input`, while high layers generally show SID/text p99 ratios below 1. This suggests the recommendation-token modality gap, if useful, is position/module-specific rather than a uniform SID-token activation scale shift.
