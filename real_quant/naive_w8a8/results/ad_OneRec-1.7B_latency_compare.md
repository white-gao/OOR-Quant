baseline_backend: hf_full_precision
candidate_backend: hf_real_naive_w8a8_scaled_mm
num_samples: 1000 vs 1000

| field | baseline | candidate | speedup/ratio |
| --- | ---: | ---: | ---: |
| generate_time_total | 541.634975 | 651.489388 | 0.8314x |
| end_to_end_time_total | 543.858137 | 653.693876 | 0.8320x |
| generated_tokens_per_generate_second | 177.241139 | 147.354664 | 0.8314x |
| samples_per_end_to_end_second | 1.838715 | 1.529768 | 0.8320x |
