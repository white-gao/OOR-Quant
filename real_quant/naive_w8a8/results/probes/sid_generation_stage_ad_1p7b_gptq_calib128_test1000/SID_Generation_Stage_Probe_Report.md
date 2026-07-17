# SID Generation Stage Probe Report

## Setup

Plain GPTQ FP8 weights are calibrated once and shared by all variants. A rescue variant uses W8A16 only at its selected autoregressive stage; all other Linear calls remain W8A8.

## End-to-End Metrics

| Variant | Pass@1 | Pass@16 | Pass@32 | Recall@32 |
| --- | ---: | ---: | ---: | ---: |
| w8a8 | 0.018000 | 0.116000 | 0.172000 | 0.058266 |
| rescue_a | 0.017000 | 0.132000 | 0.184000 | 0.061013 |
| rescue_b | 0.018000 | 0.120000 | 0.177000 | 0.061379 |
| rescue_c | 0.016000 | 0.114000 | 0.179000 | 0.059652 |
| rescue_all | 0.014000 | 0.128000 | 0.182000 | 0.064141 |

## Paired Recovery versus W8A8

| Variant | Recovery | Regression | Net Gain |
| --- | ---: | ---: | ---: |
| rescue_a | 33 | 21 | 12 |
| rescue_b | 23 | 18 | 5 |
| rescue_c | 10 | 3 | 7 |
| rescue_all | 28 | 18 | 10 |

## Returned-Beam Prefix Proxy

### rescue_a

| Stage | W8A8 coverage | Variant coverage | Delta | Prefix Jaccard |
| --- | ---: | ---: | ---: | ---: |
| A | 0.570312 | 0.562500 | -0.007812 | 0.746773 |
| B | 0.296875 | 0.281250 | -0.015625 | 0.634998 |
| C | 0.203125 | 0.218750 | 0.015625 | 0.579789 |

### rescue_b

| Stage | W8A8 coverage | Variant coverage | Delta | Prefix Jaccard |
| --- | ---: | ---: | ---: | ---: |
| A | 0.570312 | 0.570312 | 0.000000 | 0.833930 |
| B | 0.296875 | 0.296875 | 0.000000 | 0.731534 |
| C | 0.203125 | 0.210938 | 0.007812 | 0.661787 |

### rescue_c

| Stage | W8A8 coverage | Variant coverage | Delta | Prefix Jaccard |
| --- | ---: | ---: | ---: | ---: |
| A | 0.570312 | 0.578125 | 0.007812 | 0.956331 |
| B | 0.296875 | 0.289062 | -0.007812 | 0.941159 |
| C | 0.203125 | 0.218750 | 0.015625 | 0.853974 |

### rescue_all

| Stage | W8A8 coverage | Variant coverage | Delta | Prefix Jaccard |
| --- | ---: | ---: | ---: | ---: |
| A | 0.570312 | 0.562500 | -0.007812 | 0.772712 |
| B | 0.296875 | 0.289062 | -0.007812 | 0.688026 |
| C | 0.203125 | 0.226562 | 0.023438 | 0.645345 |

## Interpretation

`rescue_a` has the largest paired recovery net gain. The next justified experiment is weight-versus-activation attribution for its corresponding stage, before changing the GPTQ Hessian objective.

## Limitations

- The rescue is W8A16: it isolates activation FP8-QDQ while retaining GPTQ FP8-QDQ weights.
- Returned-beam prefix coverage is derived from final returned beams and is not an exact count of every live intermediate beam.
- This phase cannot distinguish BF16 weight rescue; run the fake-quant attribution phase only if a stage has a stable activation-rescue gain.
