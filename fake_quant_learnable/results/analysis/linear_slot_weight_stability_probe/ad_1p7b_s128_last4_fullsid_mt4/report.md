# Linear-Wise Slot Weight Split-Half Stability Probe

- task: `ad`
- split: `calib`
- sample_size: `128`; halves: `{'half_a': 64, 'half_b': 64}`
- layers: `[24, 25, 26, 27]`
- sensitivity: `mean_t ||dL/dY_{layer,linear,t}||_2^2, normalized across slot groups`

The probe estimates a separate normalized slot-group weight vector for every Linear in two disjoint calibration halves. It evaluates estimator stability, not quantization accuracy.

## Aggregate Stability

- valid Linears: `28/28`; invalid due to missing group coverage: `0`

| module type | Linears | mean cosine | median cosine | min cosine | mean Spearman | top-1 agreement | mean top-2 overlap | mean L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 28 | 0.999813 | 1.000000 | 0.998443 | 0.960714 | 0.928571 | 0.982143 | 0.012662 |
| down_proj | 4 | 0.999419 | 0.999617 | 0.998443 | 0.975000 | 0.750000 | 1.000000 | 0.028517 |
| gate_proj | 4 | 0.999993 | 0.999999 | 0.999973 | 1.000000 | 1.000000 | 1.000000 | 0.005276 |
| k_proj | 4 | 0.999999 | 1.000000 | 0.999999 | 0.875000 | 1.000000 | 0.875000 | 0.002462 |
| o_proj | 4 | 0.999455 | 0.999634 | 0.998552 | 0.975000 | 0.750000 | 1.000000 | 0.027866 |
| q_proj | 4 | 0.999839 | 0.999981 | 0.999393 | 0.925000 | 1.000000 | 1.000000 | 0.018641 |
| up_proj | 4 | 0.999989 | 0.999999 | 0.999959 | 0.975000 | 1.000000 | 1.000000 | 0.005746 |
| v_proj | 4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000126 |

## Least Stable Linears

| layer | Linear | cosine | Spearman | top-1 groups | L1 |
| ---: | --- | ---: | ---: | --- | ---: |
| 24 | mlp.down_proj | 0.998443 | 0.900000 | boundary / text | 0.062226 |
| 24 | self_attn.o_proj | 0.998552 | 0.900000 | boundary / text | 0.060221 |
| 25 | mlp.down_proj | 0.999252 | 1.000000 | boundary / boundary | 0.042593 |
| 25 | self_attn.o_proj | 0.999288 | 1.000000 | boundary / boundary | 0.041751 |
| 24 | self_attn.q_proj | 0.999393 | 0.900000 | boundary / boundary | 0.057992 |
| 25 | mlp.up_proj | 0.999959 | 1.000000 | boundary / boundary | 0.015610 |
| 25 | self_attn.q_proj | 0.999962 | 0.900000 | boundary / boundary | 0.016120 |
| 24 | mlp.gate_proj | 0.999973 | 1.000000 | boundary / boundary | 0.013960 |
| 26 | self_attn.o_proj | 0.999981 | 1.000000 | boundary / boundary | 0.009491 |
| 26 | mlp.down_proj | 0.999982 | 1.000000 | boundary / boundary | 0.009249 |

A high average similarity supports testing Linear-wise weighting; it does not establish that finer weighting improves GPTAQ. The quantization ablation must still compare Linear-wise and Layer-wise weighting under the same activation-aware target and alpha.
