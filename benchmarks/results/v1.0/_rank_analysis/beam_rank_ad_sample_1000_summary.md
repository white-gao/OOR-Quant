# Beam Rank Shift Summary

This analysis compares per-sample `test_generated.json` files. `lost` means baseline hit at top-k but the comparison run missed; `gained` means the comparison run hit when baseline missed. Rank deltas treat a missing target as `k + 1`.

| comparison | samples | sid_top1_changed_pct | sid_topk_overlap_jaccard_avg | best_sid_status_lost | best_sid_status_gained | pass@32_delta | recall@32_delta | pid_top1_changed_pct | pid_topk_overlap_jaccard_avg | best_pid_status_lost | best_pid_status_gained | pid_pass@32_delta | pid_recall@32_delta | margin_delta_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp-all | 1000 | 37.000000 | 0.654438 | 19 | 16 | -0.003000 | -0.003590 | 37.000000 | 0.654900 | 20 | 17 | -0.003000 | -0.003846 | 0.030176 |
| block-linears | 1000 | 38.700000 | 0.627331 | 24 | 21 | -0.003000 | -0.002817 | 38.700000 | 0.628208 | 23 | 23 | 0.000000 | -0.002949 | 0.027548 |
