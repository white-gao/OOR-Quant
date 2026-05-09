# Weight Probe

Utilities for inspecting OneRec-1.7B weight distributions.

Run from the benchmark repo root:

```bash
bash fake_quant/weight_probe/run_weight_distribution.sh
```

Default outputs:

```text
fake_quant/weight_probe/results/v1.0/OneRec-1.7B/
  stats/weight_stats.csv
  stats/weight_stats.json
  plots/layer_00/self_attn_q_proj_weight.png
  ...
```

The default target set excludes the large embedding table and includes all other
model weights, including Linear weights, norm weights, and `model.norm.weight`.

```text
model.layers.*.*
model.norm.weight
```

Use `INCLUDE_EMBEDDING=true` to also plot `model.embed_tokens.weight`.
