# Benchmarks

This directory is trimmed for OneRec quantization and AD-domain evaluation. The original API-wrapper, Ray/vLLM multi-node runner, and standalone HF profiling entrypoints were removed from this workspace.

## Main Entrypoints

Active fake-quant/GPTQ experiments are run from `fake_quant_learnable/`, for example:

```bash
python3 -m fake_quant_learnable.run_m1_onerec_ad --mode weighted_gptq_fp8_w8a8
```


## Retained Structure

```text
benchmark/      # RecIF-Bench loaders/evaluators used by ../fake_quant_learnable
scripts/        # offline quantization and analysis scripts
results/        # selected experiment summaries
models/         # lightweight offline quantization artifacts/configs
```

## Notes

The current research path focuses on AD SID prediction. Non-AD task code may still exist under `benchmark/tasks/v1_0/` because the registry and shared evaluator utilities are still useful, but API-based tasks are not part of the active workflow.

For current experiment context, see `../fake_quant_learnable/project_note.md`; archived fake-quant notes are in `../fake_quant_learnable/support/legacy_fake_quant_notes.md`.
