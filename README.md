# OOR-Quant

OOR-Quant is a trimmed research workspace for OneRec model quantization and AD-domain evaluation.

This repository was reduced from the original OpenOneRec codebase. Training, RL, distillation, and data-construction pipelines have been removed. The remaining code focuses on:

- vLLM/llm-compressor quantized model evaluation.
- Active HuggingFace/PyTorch W8A8, GPTQ, token-weighted GPTQ, and tail1 evaluation under `fake_quant_learnable/`.
- Experiment notes and result summaries for recommendation LLM quantization.

## Layout

```text
fake_quant_learnable/ # active W8A8/GPTQ/token-weight/tail1 experiments and notes
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

Run the active OneRec fake-quant/GPTQ runner from the repository root, for example:

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad --mode weighted_gptq_fp8_w8a8
```

The product/video serial suite is available at:

```bash
GPU_ID=0 bash fake_quant_learnable/run_product_video_quant_suite.sh
```

See `fake_quant_learnable/project_note.md` for current research notes and `fake_quant_learnable/support/legacy_fake_quant_notes.md` for the archived fake-quant notes.
