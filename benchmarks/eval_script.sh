#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
eval_script.sh is deprecated in this trimmed AD-only workspace.

The old vLLM/Ray runner under scripts/ray-vllm/ and the legacy fake_quant/
entrypoints were removed. Use the current HuggingFace/PyTorch runner instead:

  python3 -m fake_quant_learnable.run_m1_onerec_ad --mode weighted_gptq_fp8_w8a8

For the product/video serial suite:

  GPU_ID=0 bash fake_quant_learnable/run_product_video_quant_suite.sh
EOF

exit 1
