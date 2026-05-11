# Benchmarks

This directory is trimmed for OneRec quantization and AD-domain evaluation. The original API-wrapper, Ray/vLLM multi-node runner, and standalone HF profiling entrypoints were removed from this workspace.

## Main Entrypoints

Run HuggingFace fake-quant AD evaluation from the repository root:

```bash
bash fake_quant/run_hf_ad_full_quant.sh
```

Run SmoothQuant-style FP8 fake quant:

```bash
bash fake_quant/smoothquant/run_smoothquant_ad.sh
```

Run layer leave-one-out sensitivity on AD sample-1000:

```bash
bash fake_quant/run_layer_sensitivity.sh
```

## Retained Structure

```text
benchmark/      # RecIF-Bench loaders/evaluators used by ../fake_quant
scripts/        # offline quantization and analysis scripts
results/        # selected experiment summaries
models/         # lightweight offline quantization artifacts/configs
```

## Notes

The current research path focuses on AD SID prediction. Non-AD task code may still exist under `benchmark/tasks/v1_0/` because the registry and shared evaluator utilities are still useful, but API-based tasks are not part of the active workflow.

For detailed experiment context, see `CODEX_README.md` and `../fake_quant/MAIN.md`.
