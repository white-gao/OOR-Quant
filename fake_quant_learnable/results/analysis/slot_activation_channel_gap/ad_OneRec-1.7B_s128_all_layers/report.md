# Slot Activation Channel Gap Probe

- task: `ad`
- split: `calib`
- sample_size: `128`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- linear_regex: `(q_proj|o_proj|gate_proj|down_proj)$`
- groups: `['text', 'sid_a', 'sid_b', 'sid_c', 'boundary']`

## Definition

```text
energy[g,c] = mean_{token in group g}(activation[token,c]^2)
```

`energy[g]` equals the token-count-normalized diagonal of the group Hessian `X_g.T @ X_g`.

## Aggregate Channel Gap

| metric | category | mean cosine distance | mean top1% overlap |
| --- | --- | ---: | ---: |
| energy | text_vs_sid | 0.486174 | 0.390398 |
| energy | sid_internal | 0.273442 | 0.521960 |
| mean_abs | text_vs_sid | 0.182347 | 0.423129 |
| mean_abs | sid_internal | 0.111891 | 0.548833 |
| max_abs | text_vs_sid | 0.177584 | 0.322199 |
| max_abs | sid_internal | 0.080903 | 0.424210 |

## Cross-Sample Stability

| avg intra cosine | avg inter cosine | avg separation |
| ---: | ---: | ---: |
| 0.953038 | 0.578801 | 0.374237 |

## Strongest Energy Gaps

| layer | module | category | cosine distance | top1% overlap |
| ---: | --- | --- | ---: | ---: |
| 7 | mlp.down_proj | text_vs_sid | 0.934525 | 0.102151 |
| 10 | mlp.down_proj | text_vs_sid | 0.931172 | 0.215054 |
| 4 | mlp.down_proj | text_vs_sid | 0.921120 | 0.048387 |
| 3 | mlp.down_proj | text_vs_sid | 0.907034 | 0.080645 |
| 11 | mlp.down_proj | text_vs_sid | 0.907030 | 0.123656 |
| 6 | mlp.down_proj | text_vs_sid | 0.900706 | 0.037634 |
| 8 | mlp.down_proj | text_vs_sid | 0.883666 | 0.064516 |
| 1 | mlp.down_proj | text_vs_sid | 0.874576 | 0.150538 |
| 0 | mlp.down_proj | text_vs_sid | 0.867818 | 0.064516 |
| 20 | mlp.down_proj | text_vs_sid | 0.867196 | 0.059140 |
| 19 | mlp.down_proj | text_vs_sid | 0.862341 | 0.048387 |
| 22 | mlp.down_proj | text_vs_sid | 0.846200 | 0.172043 |

Complete per-module results are stored in the CSV files and `channel_profiles.pt`.
