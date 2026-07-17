# Conditional Hessian Premise Probe

- task: `ad`
- split: `calib`
- samples: `128`
- layers: `[0, 4, 8, 12, 16, 20, 24, 27]`
- linears: `(q_proj|o_proj|gate_proj|down_proj)$`

## Definition

```text
A[g,r] = mean_{t in g}(Y[t,r]^2 / mean_c(Y[t,c]^2))
pi[r,g] = A[g,r] / sum_h A[h,r]
H_r = sum_g pi[r,g] H_g
```

`pi[r,:]` is computed with equal group priors. Entropy 1.0 and max-pi 0.2 indicate no slot conditioning.

## Most Conditional Linear Outputs

| layer | module | mean entropy | mean max-pi | max-pi >= 0.35 | split-half cosine | top-1 agreement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 27 | mlp.down_proj | 0.8300 | 0.4018 | 0.7900 | 0.9998 | 0.9771 |
| 0 | mlp.gate_proj | 0.9008 | 0.3599 | 0.4727 | 0.9998 | 0.9704 |
| 16 | mlp.down_proj | 0.9022 | 0.3724 | 0.5674 | 0.9997 | 0.9717 |
| 0 | self_attn.q_proj | 0.9110 | 0.3517 | 0.3984 | 0.9998 | 0.9707 |
| 24 | self_attn.q_proj | 0.9144 | 0.3636 | 0.4805 | 0.9998 | 0.9756 |
| 20 | self_attn.q_proj | 0.9176 | 0.3597 | 0.4604 | 0.9998 | 0.9712 |
| 16 | self_attn.q_proj | 0.9221 | 0.3539 | 0.4238 | 0.9998 | 0.9678 |
| 16 | mlp.gate_proj | 0.9248 | 0.3507 | 0.4186 | 0.9997 | 0.9691 |
| 0 | self_attn.o_proj | 0.9257 | 0.3140 | 0.1543 | 0.9997 | 0.9512 |
| 8 | self_attn.o_proj | 0.9350 | 0.3416 | 0.3638 | 0.9994 | 0.9575 |
| 12 | self_attn.q_proj | 0.9352 | 0.3351 | 0.3335 | 0.9997 | 0.9707 |
| 20 | mlp.gate_proj | 0.9375 | 0.3350 | 0.3415 | 0.9997 | 0.9673 |
| 16 | self_attn.o_proj | 0.9384 | 0.3363 | 0.3442 | 0.9994 | 0.9492 |
| 4 | self_attn.q_proj | 0.9396 | 0.3257 | 0.2954 | 0.9998 | 0.9639 |
| 8 | self_attn.q_proj | 0.9410 | 0.3255 | 0.2896 | 0.9998 | 0.9678 |
| 27 | self_attn.q_proj | 0.9422 | 0.3233 | 0.2881 | 0.9997 | 0.9580 |

## Decision Rule

Proceed only if slot mixtures are both non-uniform (entropy materially below 1.0 / max-pi above 0.2) and stable across halves.
The CSV contains per-channel mixtures; the PT file contains the complete matrices for later clustering.
