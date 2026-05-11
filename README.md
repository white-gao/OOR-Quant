# OOR-Quant

OOR-Quant is a trimmed research workspace for OneRec model quantization and AD-domain evaluation.

This repository was reduced from the original OpenOneRec codebase. Training, RL, distillation, and data-construction pipelines have been removed. The remaining code focuses on:

- vLLM/llm-compressor quantized model evaluation.
- HuggingFace fake-quant evaluation for OneRec AD SID prediction.
- Activation and weight probing for quantization analysis.
- Experiment notes and result summaries for recommendation LLM quantization.

## Layout

```text
fake_quant/         # HF fake quant, SmoothQuant, probes, and local results
benchmarks/
  benchmark/      # RecIF-Bench loaders and evaluators
  scripts/        # vLLM / llm-compressor quantization and analysis scripts
  results/        # experiment summaries and selected results
  models/             # lightweight offline quantization configs/artifacts

data/
  onerec_data/
    benchmark-data/   # retained AD benchmark parquet files and sid2pid maps
```

## Main Entrypoints

The old vLLM/Ray benchmark entrypoint is deprecated in this trimmed workspace.

Run HuggingFace fake-quant AD evaluation from the repository root:

```bash
bash fake_quant/run_hf_ad_full_quant.sh
```

Run SmoothQuant-style FP8 fake quant:

```bash
bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

See `benchmarks/CODEX_README.md` and `fake_quant/MAIN.md` for experiment context, results, and current research notes.
