# OneRec Benchmark Data

This directory is trimmed for quantization/evaluation work. It keeps only the benchmark parquet files used by the current AD-domain experiments.

## Layout

```text
onerec_data/
  benchmark-data/
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

The original OpenOneRec pretraining and SFT data construction scripts were removed.

## Usage

From the repository root, run fake_quant scripts directly:

```bash
bash fake_quant/run_hf_ad_full_quant.sh
```
