"""W8A8 and SmoothQuant PTQ utilities for OneRec experiments."""

from .apply import (
    BaselineQuantSummary,
    apply_baseline_w8a8,
    install_shared_input_activation_quantization,
    iter_baseline_w8a8_modules,
    iter_gptq_w8a8_modules,
    iter_smoothquant_w8a8_modules,
)
from .gptq import collect_gptq_hessians, gptaq_fp8_quantize_weight, gptq_fp8_quantize_weight, gptq_quantized_module_from_hessians
from .gradient_weights import (
    DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG,
    GradientTokenWeightConfig,
    collect_gradient_group_token_weight_batches_by_layer,
    collect_gradient_token_weight_batches_by_layer,
    normalize_gradient_token_weights,
)
from .modules import BaselineFakeQuantLinear, GPTQFakeQuantLinear, SmoothQuantFakeQuantLinear
from .quant import fp8_e4m3_qdq_forward
from .token_weights import (
    DEFAULT_PROMPT_TOKEN_WEIGHT_CONFIG,
    DEFAULT_SLOT_TOKEN_WEIGHT_CONFIG,
    PROMPT_TOKEN_GROUP_IDS,
    PROMPT_TOKEN_GROUPS,
    SLOT_TOKEN_GROUP_IDS,
    SLOT_TOKEN_GROUPS,
    PromptTokenWeightConfig,
    SlotTokenWeightConfig,
    build_prompt_slot_token_group_batches,
    build_prompt_slot_token_groups,
    build_prompt_slot_token_weight_batches,
    build_prompt_slot_token_weights,
    build_prompt_token_group_batches,
    build_prompt_token_groups,
    build_prompt_token_weight_batches,
    build_prompt_token_weights,
)
from .support.smoothquant_runtime import (
    collect_smoothquant_scales,
    fold_smoothquant_scales_inplace,
    smoothquant_quantized_module_from_scales,
)

__all__ = [
    "BaselineFakeQuantLinear",
    "BaselineQuantSummary",
    "GPTQFakeQuantLinear",
    "SmoothQuantFakeQuantLinear",
    "DEFAULT_GRADIENT_TOKEN_WEIGHT_CONFIG",
    "DEFAULT_PROMPT_TOKEN_WEIGHT_CONFIG",
    "DEFAULT_SLOT_TOKEN_WEIGHT_CONFIG",
    "GradientTokenWeightConfig",
    "PROMPT_TOKEN_GROUP_IDS",
    "PROMPT_TOKEN_GROUPS",
    "SLOT_TOKEN_GROUP_IDS",
    "SLOT_TOKEN_GROUPS",
    "PromptTokenWeightConfig",
    "SlotTokenWeightConfig",
    "apply_baseline_w8a8",
    "build_prompt_slot_token_group_batches",
    "build_prompt_slot_token_groups",
    "build_prompt_slot_token_weight_batches",
    "build_prompt_slot_token_weights",
    "build_prompt_token_group_batches",
    "build_prompt_token_groups",
    "build_prompt_token_weight_batches",
    "build_prompt_token_weights",
    "collect_gradient_group_token_weight_batches_by_layer",
    "collect_gradient_token_weight_batches_by_layer",
    "collect_gptq_hessians",
    "collect_smoothquant_scales",
    "fold_smoothquant_scales_inplace",
    "fp8_e4m3_qdq_forward",
    "gptaq_fp8_quantize_weight",
    "gptq_fp8_quantize_weight",
    "gptq_quantized_module_from_hessians",
    "install_shared_input_activation_quantization",
    "iter_baseline_w8a8_modules",
    "iter_gptq_w8a8_modules",
    "iter_smoothquant_w8a8_modules",
    "normalize_gradient_token_weights",
    "smoothquant_quantized_module_from_scales",
]
