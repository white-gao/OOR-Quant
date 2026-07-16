# Linear-Wise Slot Group Gradient Gap Probe

This report re-aggregates the existing full-SID multi-target Linear-output sensitivity probe.

For each `(layer, Linear, metric)`, group scalar energy is the sum of the mean channel profile, then normalized across:

```text
text / sid_a / sid_b / sid_c / boundary
```

The resulting profile is a diagnostic for whether a layer-shared group weight is likely too coarse. It is not yet a deployed linear-wise weighting rule.

## Same-Layer Module Gap

| metric | layers | module pairs | avg cosine | min cosine | avg L1 | max L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grad2 | 4 | 84 | 0.606659 | 0.003060 | 0.911629 | 1.993899 |
| gx | 4 | 84 | 0.842587 | 0.223036 | 0.447914 | 1.705266 |

## Average Relative Group Importance by Linear

| metric | Linear | text | sid_a | sid_b | sid_c | boundary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| grad2 | mlp.down_proj | 0.265534 | 0.007575 | 0.000863 | 0.001405 | 0.724623 |
| grad2 | mlp.gate_proj | 0.111794 | 0.004435 | 0.000920 | 0.000492 | 0.882358 |
| grad2 | mlp.up_proj | 0.097609 | 0.003632 | 0.000668 | 0.000473 | 0.897618 |
| grad2 | self_attn.k_proj | 0.995431 | 0.001872 | 0.000137 | 0.000231 | 0.002330 |
| grad2 | self_attn.o_proj | 0.262930 | 0.007726 | 0.000979 | 0.001383 | 0.726982 |
| grad2 | self_attn.q_proj | 0.060188 | 0.001269 | 0.000374 | 0.000658 | 0.937512 |
| grad2 | self_attn.v_proj | 0.997895 | 0.000065 | 0.000007 | 0.000022 | 0.002011 |
| gx | mlp.down_proj | 0.330679 | 0.102867 | 0.049962 | 0.030720 | 0.485772 |
| gx | mlp.gate_proj | 0.345547 | 0.101226 | 0.049937 | 0.029778 | 0.473512 |
| gx | mlp.up_proj | 0.345078 | 0.099589 | 0.048670 | 0.031247 | 0.475416 |
| gx | self_attn.k_proj | 0.512320 | 0.283917 | 0.064141 | 0.053826 | 0.085796 |
| gx | self_attn.o_proj | 0.365267 | 0.058803 | 0.033747 | 0.040819 | 0.501364 |
| gx | self_attn.q_proj | 0.323099 | 0.060029 | 0.040492 | 0.046170 | 0.530210 |
| gx | self_attn.v_proj | 0.499448 | 0.272248 | 0.061187 | 0.053373 | 0.113744 |

## Most Different Same-Layer Pairs (grad2)

| layer | Linear A | Linear B | cosine | L1 |
| ---: | --- | --- | ---: | ---: |
| 27 | mlp.down_proj | self_attn.k_proj | 0.003060 | 1.993899 |
| 27 | mlp.gate_proj | self_attn.k_proj | 0.003060 | 1.993899 |
| 27 | mlp.up_proj | self_attn.k_proj | 0.003060 | 1.993899 |
| 27 | self_attn.k_proj | self_attn.o_proj | 0.003060 | 1.993899 |
| 27 | self_attn.k_proj | self_attn.q_proj | 0.003060 | 1.993899 |
| 27 | mlp.down_proj | self_attn.v_proj | 0.006434 | 1.987213 |
| 27 | mlp.gate_proj | self_attn.v_proj | 0.006434 | 1.987213 |
| 27 | mlp.up_proj | self_attn.v_proj | 0.006434 | 1.987213 |

## Most Different Same-Layer Pairs (gx)

| layer | Linear A | Linear B | cosine | L1 |
| ---: | --- | --- | ---: | ---: |
| 27 | mlp.down_proj | self_attn.k_proj | 0.223036 | 1.705266 |
| 27 | mlp.gate_proj | self_attn.k_proj | 0.223036 | 1.705266 |
| 27 | mlp.up_proj | self_attn.k_proj | 0.223036 | 1.705266 |
| 27 | self_attn.k_proj | self_attn.o_proj | 0.223036 | 1.705266 |
| 27 | self_attn.k_proj | self_attn.q_proj | 0.223036 | 1.705266 |
| 27 | mlp.down_proj | self_attn.v_proj | 0.322520 | 1.587447 |
| 27 | mlp.gate_proj | self_attn.v_proj | 0.322520 | 1.587447 |
| 27 | mlp.up_proj | self_attn.v_proj | 0.322520 | 1.587447 |
