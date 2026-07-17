# SID Generation Stage Probe

## Question

The existing token-gap probes establish that text and SID tokens have different
representations and activation-channel patterns. They do not establish which
positions should be protected during quantization, nor whether the relevant
error is caused by weights or activations.

This probe tests the causal chain:

```text
quantization at a SID prediction stage
  -> beam-margin degradation / prefix pruning
  -> recommendation failure
  -> metric recovery when that stage is restored
```

The baseline is Plain GPTQ W8A8. Its quantized weights remain fixed in every
activation-rescue variant.

## Stage Definitions

For three-token Semantic IDs, an autoregressive prediction stage is defined by
the hidden state that predicts the *next* SID token:

| Stage | Input position processed by the model | Next token predicted |
| --- | --- | --- |
| A | Final `<|sid_begin|>` position in prefill | `SID-a` |
| B | First generated `SID-a` token in decode | `SID-b` |
| C | Second generated `SID-b` token in decode | `SID-c` |

This differs from grouping every occurrence of `SID-a`, `SID-b`, or `SID-c` in
the prompt. History tokens and target-generation positions have different
causal roles.

## Experiment Sequence

### 1. Beam trajectory observation

Run BF16 and Plain GPTQ W8A8 with the same beam configuration. For every
sample and stage, record generated candidate prefixes and score summaries.
The key outcome is the first generation stage at which the two models' beam
trajectories diverge.

### 2. Stage-wise activation rescue

The following variants use the same GPTQ FP8 weights. The selected stage uses
W8A16: activations are kept in BF16 while the already-quantized QDQ weight is
used. The other stages remain W8A8.

| Variant | Stage A | Stage B | Stage C |
| --- | --- | --- | --- |
| `w8a8` | W8A8 | W8A8 | W8A8 |
| `rescue_a` | W8A16 | W8A8 | W8A8 |
| `rescue_b` | W8A8 | W8A16 | W8A8 |
| `rescue_c` | W8A8 | W8A8 | W8A16 |
| `rescue_all` | W8A16 | W8A16 | W8A16 |

Stage A is implemented by splitting only the final prefill token in each
`RealFP8Linear`; it does not convert the entire prompt prefill to BF16.
Stages B and C are the first and second `seq_len=1` decode forwards.

For each rescue variant, compare against the paired W8A8 sample result:

```text
recovery:   W8A8 fails and rescue succeeds
regression: W8A8 succeeds and rescue fails
net gain:   recovery - regression
```

The primary metric is Pass@32 / Recall@32. Paired recovery counts are more
sensitive than aggregate metrics on test1000.

### 3. Weight versus activation attribution

Only run this phase for a stage that has a stable activation-rescue benefit.
The fake-quant probe will retain original BF16 weights and compare:

| Mode | Weight path | Activation path | Interpretation |
| --- | --- | --- | --- |
| Base | FP8 QDQ | FP8 QDQ | ordinary W8A8 |
| A16 rescue | FP8 QDQ | BF16 | activation error contribution |
| W16 rescue | BF16 | FP8 QDQ | weight error contribution |
| WA16 rescue | BF16 | BF16 | total stage quantization upper bound |

### 4. Optional layer localization

Only if a stage is important and weight rescue has a stable benefit, repeat the
rescue over four layer ranges: `0-6`, `7-13`, `14-20`, `21-27`. This prevents a
subsequent Hessian method from weighting all layers without evidence.

## Decision Rules

| Observation | Consequence |
| --- | --- |
| One stage has a clear paired recovery gain | Treat that prediction stage as a candidate quantization bottleneck |
| Only A16 rescue helps | Prioritize activation-side quantization / smoothing, not Hessian reweighting |
| W16 rescue helps beyond A16 | A stage-aware GPTQ objective is justified for that stage |
| All-stage rescue helps but individual stages do not | Error is distributed or cross-stage coupled |
| No rescue has a stable effect | Do not claim a SID stage should receive special protection |

## Implementation Isolation

The runtime uses a context-local stage label enabled only by the probe runner.
Normal `real_quant` commands have no stage label and preserve their prior
W8A8, tail-token, and decode-A16 behavior.
