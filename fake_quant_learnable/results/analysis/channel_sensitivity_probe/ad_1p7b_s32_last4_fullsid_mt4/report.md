# Channel Sensitivity Probe

- task: `ad`
- split: `calib`
- sample_size: `32`
- layers: `[24, 25, 26, 27]`
- linear_regex: `(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`
- target: `multi_target_full_sid_teacher_forcing`

## How to Read

`grad2` 对齐 GuidedQuant 中 Hessian token 权重的来源；`gx` 是 `|activation * gradient|` 的辅助敏感度。slot pair similarity 比较不同 slot 的平均 channel profile；token-level consistency 比较同一 slot 内 token profile 是否比跨 slot 更相似。

## Aggregate Signals

| metric | avg cv | avg top32 share | max top32 share |
| --- | ---: | ---: | ---: |
| grad2 | 2.949019 | 0.226707 | 0.591865 |
| gx | 1.352681 | 0.130354 | 0.377087 |

## Slot Pair Similarity

| metric | avg cosine | min cosine | avg top32 overlap | min top32 overlap |
| --- | ---: | ---: | ---: | ---: |
| grad2 | 0.514616 | 0.048343 | 0.267527 | 0.000000 |
| gx | 0.741457 | 0.313697 | 0.368750 | 0.000000 |

## Token-Level Slot Consistency

| metric | avg intra cosine | avg inter cosine | avg separation | avg intra top32 overlap | avg inter top32 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| grad2 | 0.354276 | 0.257364 | 0.096912 | 0.209809 | 0.135184 |
| gx | 0.420559 | 0.315184 | 0.105375 | 0.201636 | 0.122856 |

完整逐层结果见 `channel_profile_summary.csv`、`channel_profile_similarity.csv` 和 `channel_token_consistency.csv`。
