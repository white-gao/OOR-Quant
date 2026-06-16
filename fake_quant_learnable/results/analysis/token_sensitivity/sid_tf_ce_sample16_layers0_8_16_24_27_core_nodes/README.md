# SID Token Sensitivity Probe

This probe is a lightweight MBQ-style validation for OneRec-1.7B.
It uses the FP model and teacher-forced ground-truth SID CE loss, then backpropagates gradients to selected hidden states.

## Command

```bash
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m fake_quant.probes.token_sensitivity.probe_sid_token_sensitivity   --model_path /home/guowei/OneRec-1.7B/   --data_dir data/onerec_data/benchmark-data-calib1024   --split calib   --sample_size 16   --sample_offset 0   --device cuda:4   --layers 0,8,16,24,27   --nodes attn_qkv_input,ffn_gate_up_input,block_output   --output_dir fake_quant_learnable/results/analysis/token_sensitivity/sid_tf_ce_sample16_layers0_8_16_24_27_core_nodes
```

## Metric

For each layer/node/token group, the script reports:

```text
mean_abs_grad = mean(|dL/dY|)
mean_abs_act_grad = mean(|Y| * |dL/dY|)
ratio_to_text = value / text_prompt_value at the same layer/node
```

`mean_abs_act_grad` is the closer MBQ-style first-order sensitivity proxy.

## Main Observation

The teacher-forcing SID prediction positions are much more sensitive than ordinary prompt text tokens.
Averaged over non-degenerate layer/node entries:

```text
predict_s_b_position: act_grad/text ~= 91.7x
predict_s_c_position: act_grad/text ~= 57.5x
predict_s_a_position: act_grad/text ~= 51.4x
text_prompt:          act_grad/text = 1.0x
history_sid_a:        act_grad/text ~= 0.40x
history_sid_b:        act_grad/text ~= 0.25x
history_sid_c:        act_grad/text ~= 0.19x
history_sid_boundary: act_grad/text ~= 0.12x
```

This suggests that token sensitivity exists, but the strongest signal is not a broad SID-history-vs-text modality gap.
Instead, the supervised SID decode positions are the dominant sensitive tokens.

## Files

```text
summary.json              full per-layer/node/group statistics
summary.csv               flat per-layer/node/group table
group_ratio_summary.csv   aggregated token-group ratios
```

## Caveat

The final layer `block_output` has zero gradient for history/text tokens because no later token mixing happens after that output. Use attention/FFN inputs or earlier-layer outputs when interpreting history-token sensitivity.
