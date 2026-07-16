# Token-Wise Slot Activation Outlier Probe

- task: `ad`
- split: `calib`
- sample_size: `128`
- calibration halves: `{'0': 64, '1': 64}`
- per-token top fraction: `2%`
- representative sample index: `3`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`

## Definition

```text
C_t = TopK_channel(abs(X[t,c]), K)
count[g,c] = sum_{t in group g} 1[c in C_t]
frequency[g,c] = count[g,c] / N_g

intra[g] = sum_c choose(count[g,c], 2) / (choose(N_g, 2) * K)
inter[g,h] = sum_c count[g,c] * count[h,c] / (N_g * N_h * K)
```

Every token selects its channels before any group aggregation. Intra-group and inter-group overlaps are exact averages over token pairs, computed from occurrence counts without pair sampling.

## Aggregate Result

| metric | value |
| --- | ---: |
| random top-k overlap baseline | 0.020020 |
| within-group token overlap | 0.302367 |
| text-vs-SID token overlap | 0.200020 |
| SID-internal token overlap | 0.239411 |
| split-half frequency cosine | 0.999265 |
| recurrent-channel concentration | 0.436654 |

Within-group overlap exceeds text-vs-SID overlap at `112/112` layer-module positions.

Within-group overlap exceeds SID-internal overlap at `111/112` layer-module positions.

## By Token Group

| group | intra-token overlap | split frequency cosine | recurrent concentration |
| --- | ---: | ---: | ---: |
| `text` | 0.270054 | 0.999704 | 0.414259 |
| `sid_a` | 0.320012 | 0.999037 | 0.445707 |
| `sid_b` | 0.280534 | 0.999146 | 0.407654 |
| `sid_c` | 0.338870 | 0.999174 | 0.478996 |

## By Module

| module | within group | text-vs-SID | SID-internal | intra - text/SID |
| --- | ---: | ---: | ---: | ---: |
| `self_attn.q_proj` | 0.430672 | 0.344552 | 0.367827 | 0.086120 |
| `self_attn.o_proj` | 0.277785 | 0.149459 | 0.205714 | 0.128326 |
| `mlp.gate_proj` | 0.323204 | 0.232897 | 0.262356 | 0.090307 |
| `mlp.down_proj` | 0.177809 | 0.073170 | 0.121747 | 0.104639 |

## Pair-Wise Token Overlap

| pair | inter-token overlap | pair intra reference | intra - inter | frequency cosine |
| --- | ---: | ---: | ---: | ---: |
| `sid_a - boundary` | 0.224497 | 0.373719 | 0.149222 | 0.587768 |
| `sid_a - sid_b` | 0.261159 | 0.300273 | 0.039114 | 0.860645 |
| `sid_a - sid_c` | 0.230102 | 0.329441 | 0.099339 | 0.684346 |
| `sid_b - boundary` | 0.209887 | 0.353980 | 0.144094 | 0.576580 |
| `sid_b - sid_c` | 0.226972 | 0.309702 | 0.082731 | 0.724612 |
| `sid_c - boundary` | 0.266187 | 0.383148 | 0.116961 | 0.666415 |
| `text - boundary` | 0.235112 | 0.348740 | 0.113628 | 0.645150 |
| `text - sid_a` | 0.193488 | 0.295033 | 0.101545 | 0.615970 |
| `text - sid_b` | 0.191527 | 0.275294 | 0.083767 | 0.646341 |
| `text - sid_c` | 0.215044 | 0.304462 | 0.089418 | 0.664952 |

The representative token-channel figures are qualitative. All overlap and frequency statistics use every valid calibration token.
