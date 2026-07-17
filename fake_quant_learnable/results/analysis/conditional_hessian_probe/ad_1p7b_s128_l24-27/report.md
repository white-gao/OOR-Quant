# Conditional Hessian Premise Probe

- task: `ad`
- split: `calib`
- samples: `128`
- layers: `[24, 25, 26, 27]`
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
| 24 | self_attn.q_proj | 0.9144 | 0.3636 | 0.4805 | 0.9998 | 0.9756 |
| 25 | self_attn.q_proj | 0.9164 | 0.3604 | 0.4756 | 0.9998 | 0.9785 |
| 26 | self_attn.q_proj | 0.9250 | 0.3474 | 0.3979 | 0.9997 | 0.9658 |
| 27 | self_attn.q_proj | 0.9422 | 0.3233 | 0.2881 | 0.9997 | 0.9580 |
| 26 | mlp.down_proj | 0.9472 | 0.3008 | 0.1382 | 0.9996 | 0.8882 |
| 24 | self_attn.o_proj | 0.9499 | 0.3199 | 0.2520 | 0.9992 | 0.9390 |
| 26 | mlp.gate_proj | 0.9500 | 0.3122 | 0.2100 | 0.9996 | 0.9489 |
| 27 | mlp.gate_proj | 0.9512 | 0.3082 | 0.1914 | 0.9997 | 0.9539 |
| 25 | mlp.gate_proj | 0.9531 | 0.3094 | 0.1945 | 0.9997 | 0.9487 |
| 24 | mlp.gate_proj | 0.9550 | 0.3065 | 0.1859 | 0.9997 | 0.9582 |
| 25 | mlp.down_proj | 0.9604 | 0.2817 | 0.0586 | 0.9996 | 0.9292 |
| 24 | mlp.down_proj | 0.9616 | 0.2786 | 0.0557 | 0.9996 | 0.8857 |
| 25 | self_attn.o_proj | 0.9629 | 0.3023 | 0.1753 | 0.9988 | 0.9160 |
| 26 | self_attn.o_proj | 0.9656 | 0.2896 | 0.1104 | 0.9989 | 0.8853 |
| 27 | self_attn.o_proj | 0.9671 | 0.2916 | 0.1245 | 0.9994 | 0.9302 |

## Decision Rule

Proceed only if slot mixtures are both non-uniform (entropy materially below 1.0 / max-pi above 0.2) and stable across halves.
The CSV contains per-channel mixtures; the PT file contains the complete matrices for later clustering.
