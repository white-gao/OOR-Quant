baseline_backend: hf_full_precision
candidate_backend: hf_real_naive_w8a8_scaled_mm
num_samples: 100 vs 100

| field | baseline | candidate | speedup/ratio |
| --- | ---: | ---: | ---: |
| generate_time_total | 25.886059 | 44.422320 | 0.5827x |
| end_to_end_time_total | 26.128275 | 44.683866 | 0.5847x |
| generated_tokens_per_generate_second | 370.855991 | 216.107576 | 0.5827x |
| samples_per_end_to_end_second | 3.827271 | 2.237944 | 0.5847x |
