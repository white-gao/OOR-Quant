# Prefill SID Sensitivity Probe

This directory visualizes per-token prefill sensitivity for one OneRec AD sample.

Loss:

```text
CE(logits at the final <|sid_begin|> position, ground-truth s_a)
```

Sensitivity:

```text
mean_channel(|activation| * |dLoss/dactivation|)
```

Sample ID: `0`
Target: `<s_a_4495>`
Prompt tokens: `781`

The plots use token index on the x-axis and sensitivity on the y-axis. SID-code tokens are color-marked by group; the final `<|sid_begin|>` is marked as `predict_s_a_position`.
