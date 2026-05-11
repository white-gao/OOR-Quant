#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
eval_script.sh is deprecated in this trimmed AD-only workspace.

The old vLLM/Ray runner under scripts/ray-vllm/ was removed. Use the current
HuggingFace fake-quant entrypoints instead:

  bash fake_quant/run_hf_ad_full_quant.sh

For SmoothQuant:

  bash fake_quant/smoothquant/run_smoothquant_ad.sh
EOF

exit 1
