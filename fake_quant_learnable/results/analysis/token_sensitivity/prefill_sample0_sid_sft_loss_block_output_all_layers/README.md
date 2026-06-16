# Prefill SID Sensitivity Probe

This directory visualizes per-token prefill sensitivity for one OneRec AD sample.

Loss:

```text
mean CE for teacher-forced SID tokens: <|sid_begin|>->s_a, s_a->s_b, s_b->s_c
```

Sensitivity:

```text
mean_channel(|activation| * |dLoss/dactivation|)
```

Sample ID: `0`
Target: `<s_a_4495><s_b_6857><s_c_7947>`
Input tokens: `783`
Loss mode: `sid_sft`

The plots use token index on the x-axis and sensitivity on the y-axis. SID-code tokens are color-marked by group; teacher-forced prediction positions are marked as `predict_s_*_position`.
