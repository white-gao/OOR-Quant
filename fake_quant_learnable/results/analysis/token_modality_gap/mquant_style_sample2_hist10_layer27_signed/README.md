# Token Modality Activation Gap Probe

This probe follows the idea of MQuant Figure 1(b): compare activation distributions from two token modalities. Here, OneRec history SID code tokens (`<s_a_*>`, `<s_b_*>`, `<s_c_*>`) are treated as recommendation-modality tokens, and ordinary prompt tokens are treated as text tokens. SID boundary and chat special tokens are excluded.

Run command:

> Legacy reproduction note: this command referenced the removed `fake_quant/` probe package. Keep this README as historical provenance; rerunning requires restoring or rewriting the old probe.

```bash
python3 -m fake_quant.probes.activation_probe.plot_token_modality_distribution \
  --device cuda:6 \
  --sample_size 2 \
  --sample_offset 0 \
  --max_history_sid_items 10 \
  --layers 27 \
  --nodes attn_qkv_input,ffn_gate_up_input,ffn_down_input \
  --capture_max_values_per_modality 5000 \
  --plot_max_values 20000 \
  --skip_abs_tail \
  --output_dir fake_quant_learnable/results/analysis/token_modality_gap/mquant_style_sample2_hist10_layer27_signed
```

Token counts after prompt compression:

| sample | prompt tokens | text tokens | SID code tokens | ignored tokens | kept/original SID items |
|---|---:|---:|---:|---:|---:|
| 0 | 125 | 67 | 30 | 28 | 10/120 |
| 1 | 126 | 68 | 30 | 28 | 10/101 |

Layer 27 SID/text activation magnitude ratios:

| node | mean_abs | p99_abs | p999_abs | absmax |
|---|---:|---:|---:|---:|
| attn_qkv_input | 0.803 | 0.875 | 0.588 | 0.510 |
| ffn_gate_up_input | 0.780 | 0.875 | 0.457 | 0.363 |
| ffn_down_input | 1.006 | 0.814 | 1.043 | 1.709 |

Initial observation: unlike the MLLM visual-token case described by MQuant, SID code tokens do not show a globally wider or consistently larger activation distribution than text tokens in this small prefill sample. The only clear SID-over-text tail signal here is the extreme absmax at `ffn_down_input`; qkv and gate/up inputs are mostly smaller for SID tokens.
