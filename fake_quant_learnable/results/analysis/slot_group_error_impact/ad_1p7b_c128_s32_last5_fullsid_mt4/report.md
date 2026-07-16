# Slot Group Error-Impact Probe

This probe quantizes one Transformer block at a time with plain GPTQ W8A8. For each prompt group, it measures the block-output reconstruction error and injects only that group's true quantization residual into the BF16 model before evaluating full-SID teacher-forcing CE.

- Calibration: `128` samples from `calib`
- Probe: `32` samples, full-SID multi-target CE with up to `4` targets
- Layers: `[23, 24, 25, 26, 27]`

## Layer Summary

| Layer | Group | Rel. reconstruction error | CE delta after residual injection | Tokens/sample |
|---:|---|---:|---:|---:|
| 23 | text | 0.000270236 | 0.000599794 | 248.88 |
| 23 | sid_a | 0.000334912 | 0.00248472 | 442.22 |
| 23 | sid_b | 0.000658665 | 0.00101221 | 442.22 |
| 23 | sid_c | 8.57266e-05 | 0.00182379 | 442.22 |
| 23 | boundary | 6.95744e-05 | 0.00032863 | 887.59 |
| 24 | text | 0.000285349 | 0.00183056 | 248.88 |
| 24 | sid_a | 0.000317798 | 0.00459858 | 442.22 |
| 24 | sid_b | 0.000603857 | 0.00075049 | 442.22 |
| 24 | sid_c | 9.18166e-05 | 0.000675738 | 442.22 |
| 24 | boundary | 8.26248e-05 | -0.00137072 | 887.59 |
| 25 | text | 0.000352166 | 0.00318197 | 248.88 |
| 25 | sid_a | 0.000336968 | 0.000465877 | 442.22 |
| 25 | sid_b | 0.000700778 | -0.000895195 | 442.22 |
| 25 | sid_c | 0.000133139 | -0.00195283 | 442.22 |
| 25 | boundary | 0.000110818 | -0.000674509 | 887.59 |
| 26 | text | 0.000327742 | 0.00185406 | 248.88 |
| 26 | sid_a | 0.000549986 | -0.000169054 | 442.22 |
| 26 | sid_b | 0.00105117 | 0.000484496 | 442.22 |
| 26 | sid_c | 0.000124894 | 0.000641778 | 442.22 |
| 26 | boundary | 7.64893e-05 | 0.00157523 | 887.59 |
| 27 | text | 0.000756718 | 0 | 248.88 |
| 27 | sid_a | 0.00102767 | 0 | 442.22 |
| 27 | sid_b | 0.00166297 | 0 | 442.22 |
| 27 | sid_c | 0.00069393 | 0 | 442.22 |
| 27 | boundary | 0.000363127 | 0.00140138 | 887.59 |

Interpretation: a low-frequency SID group with high relative reconstruction error and positive CE delta is evidence that frequency-weighted global reconstruction can under-protect that group.
