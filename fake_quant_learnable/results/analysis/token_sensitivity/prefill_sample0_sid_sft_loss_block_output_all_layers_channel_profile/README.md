# Full Token-Channel SID Sensitivity

This directory contains full token-channel sensitivity visualizations for one OneRec AD sample.

Loss:

```text
mean CE for teacher-forced SID tokens: <|sid_begin|>->s_a, s_a->s_b, s_b->s_c
```

Matrix definition:

```text
S[token, channel] = |activation[token, channel]| * |dLoss/dactivation[token, channel]|
```

No token or channel filtering is applied in the heatmaps. The heatmap color is `log10(S + eps)` only for readability.

Sample ID: `0`
Target: `<s_a_4495><s_b_6857><s_c_7947>`
Input tokens: `783`
Loss mode: `sid_sft`
Layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
Nodes: `['block_output']`

Files:

```text
token_channel_heatmap__*.png   full token x channel heatmaps
channel_profile__*.png         channel-wise profiles averaged by token group
channel_profile_summary.csv    numeric profile summary
token_metadata.json            token index/group/label metadata
summary.json                   run config and plot paths
```
