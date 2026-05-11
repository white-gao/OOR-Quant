# Data

This directory now only keeps benchmark data needed by the quantization/evaluation workflow.

## Layout

```text
data/onerec_data/benchmark-data/
  ad/
    ad_test.parquet
    ad_test_sample_1.parquet
    ad_test_sample_10.parquet
    ad_test_sample_100.parquet
    ad_test_sample_128.parquet
    ad_test_sample_1000.parquet
    sid2pid.json
  sid2pid.json
```

The original OpenOneRec pretraining, SFT, distillation, RL, and general-text data construction scripts were removed during repository cleanup.

## Usage

Current fake_quant scripts are run from the repository root and default to this data directory.

```bash
python fake_quant/run_ad_sid.py \
  --data_dir data/onerec_data/benchmark-data \
  --sample_size 1000 \
  --evaluate
```

Current scripts default to `data/onerec_data/benchmark-data` after the cleanup.
