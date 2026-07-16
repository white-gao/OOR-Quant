# Token-Wise Slot Activation Outlier Probe

- task: `ad`
- split: `calib`
- sample_size: `128`
- calibration halves: `{'0': 64, '1': 64}`
- per-token top fraction: `1%`
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
| random top-k overlap baseline | 0.010213 |
| within-group token overlap | 0.320936 |
| text-vs-SID token overlap | 0.205304 |
| SID-internal token overlap | 0.246502 |
| split-half frequency cosine | 0.999202 |
| recurrent-channel concentration | 0.453903 |

Within-group overlap exceeds text-vs-SID overlap at `112/112` layer-module positions.

Within-group overlap exceeds SID-internal overlap at `112/112` layer-module positions.

## By Token Group

| group | intra-token overlap | split frequency cosine | recurrent concentration |
| --- | ---: | ---: | ---: |
| `text` | 0.295827 | 0.999667 | 0.432652 |
| `sid_a` | 0.322737 | 0.998936 | 0.450169 |
| `sid_b` | 0.295281 | 0.999038 | 0.424536 |
| `sid_c` | 0.369898 | 0.999166 | 0.508254 |

## By Module

| module | within group | text-vs-SID | SID-internal | intra - text/SID |
| --- | ---: | ---: | ---: | ---: |
| `self_attn.q_proj` | 0.471805 | 0.354182 | 0.393739 | 0.117623 |
| `self_attn.o_proj` | 0.256840 | 0.124020 | 0.180202 | 0.132820 |
| `mlp.gate_proj` | 0.396654 | 0.289438 | 0.315824 | 0.107216 |
| `mlp.down_proj` | 0.158444 | 0.053577 | 0.096244 | 0.104867 |

## Pair-Wise Token Overlap

| pair | inter-token overlap | pair intra reference | intra - inter | frequency cosine |
| --- | ---: | ---: | ---: | ---: |
| `sid_a - boundary` | 0.223447 | 0.379876 | 0.156428 | 0.553666 |
| `sid_a - sid_b` | 0.263358 | 0.309009 | 0.045651 | 0.835524 |
| `sid_a - sid_c` | 0.238016 | 0.346317 | 0.108302 | 0.652160 |
| `sid_b - boundary` | 0.209299 | 0.366147 | 0.156848 | 0.533862 |
| `sid_b - sid_c` | 0.238133 | 0.332589 | 0.094456 | 0.688088 |
| `sid_c - boundary` | 0.276592 | 0.403456 | 0.126864 | 0.637391 |
| `text - boundary` | 0.241515 | 0.366421 | 0.124906 | 0.603276 |
| `text - sid_a` | 0.194832 | 0.309282 | 0.114450 | 0.561552 |
| `text - sid_b` | 0.195497 | 0.295554 | 0.100057 | 0.588781 |
| `text - sid_c` | 0.225583 | 0.332862 | 0.107279 | 0.611131 |

The representative token-channel figures are qualitative. All overlap and frequency statistics use every valid calibration token.
