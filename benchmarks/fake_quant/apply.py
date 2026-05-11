from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import re
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn

from .modules import FakeQuantLinear
from .quant import ActQuant, ActQuantMode, fp8_activation_per_token
from .smoothquant.core import (
    SmoothQuantLinear,
    compute_smooth_scales_for_model,
    load_activation_absmax,
)


@dataclass
class FakeQuantSummary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def apply_fp8_fake_quant(
    model: nn.Module,
    *,
    act_quant: ActQuant = "none",
    act_quant_mode: ActQuantMode = "per_linear",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> FakeQuantSummary:
    """Replace Linear modules with FP8 fake-quant Linear wrappers."""
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")
    if act_quant == "none" and act_quant_mode != "per_linear":
        raise ValueError("act_quant_mode is only meaningful when act_quant is enabled.")

    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    linear_act_quant = "none" if act_quant_mode == "shared_input" else act_quant
    replaced, skipped = _replace_children(
        module=model,
        prefix="",
        act_quant=linear_act_quant,
        skip_module_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
    )
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = _install_shared_input_qdq(model)

    return FakeQuantSummary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def apply_smoothquant_fp8_fake_quant(
    model: nn.Module,
    *,
    activation_absmax_path: str,
    alpha: float = 0.5,
    act_quant: ActQuant = "none",
    act_quant_mode: ActQuantMode = "per_linear",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> FakeQuantSummary:
    """Apply SmoothQuant-style smoothing plus FP8 fake-quant Linear wrappers."""
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")
    if act_quant == "none" and act_quant_mode != "per_linear":
        raise ValueError("act_quant_mode is only meaningful when act_quant is enabled.")

    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    activation_absmax = load_activation_absmax(activation_absmax_path)
    smooth_scales = compute_smooth_scales_for_model(
        model,
        activation_absmax,
        alpha=alpha,
        skip_module_names=skip_names,
        target_regex=target_regex,
        skip_regex=skip_regex,
    )

    linear_act_quant = "none" if act_quant_mode == "shared_input" else act_quant
    smooth_input = act_quant_mode != "shared_input"
    replaced, skipped = _replace_children_smoothquant(
        module=model,
        prefix="",
        act_quant=linear_act_quant,
        smooth_input=smooth_input,
        smooth_scales=smooth_scales,
        skip_module_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
    )

    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = _install_shared_input_qdq(model)

    return FakeQuantSummary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def _replace_children(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    skip_module_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> Tuple[int, int]:
    replaced = 0
    skipped = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, FakeQuantLinear):
            continue

        if isinstance(child, nn.Linear):
            if child_name in skip_module_names or full_name in skip_module_names:
                skipped += 1
                continue
            if skip_pattern is not None and skip_pattern.search(full_name) is not None:
                skipped += 1
                continue
            if target_pattern is not None and target_pattern.search(full_name) is None:
                skipped += 1
                continue
            setattr(module, child_name, FakeQuantLinear(child, act_quant=act_quant))
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children(
            module=child,
            prefix=full_name,
            act_quant=act_quant,
            skip_module_names=skip_module_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def _replace_children_smoothquant(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    smooth_input: bool,
    smooth_scales: dict[str, torch.Tensor],
    skip_module_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> Tuple[int, int]:
    replaced = 0
    skipped = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, (FakeQuantLinear, SmoothQuantLinear)):
            continue

        if isinstance(child, nn.Linear):
            if child_name in skip_module_names or full_name in skip_module_names:
                skipped += 1
                continue
            if skip_pattern is not None and skip_pattern.search(full_name) is not None:
                skipped += 1
                continue
            if target_pattern is not None and target_pattern.search(full_name) is None:
                skipped += 1
                continue
            if full_name not in smooth_scales:
                raise KeyError(f"Missing SmoothQuant scale for Linear module: {full_name}")
            setattr(
                module,
                child_name,
                SmoothQuantLinear(
                    child,
                    smooth_scale=smooth_scales[full_name],
                    act_quant=act_quant,
                    smooth_input=smooth_input,
                ),
            )
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children_smoothquant(
            module=child,
            prefix=full_name,
            act_quant=act_quant,
            smooth_input=smooth_input,
            smooth_scales=smooth_scales,
            skip_module_names=skip_module_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def _install_shared_input_qdq(model: nn.Module) -> Tuple[int, int]:
    attention_modules = 0
    mlp_modules = 0

    for module in model.modules():
        if _is_qwen3_attention_like(module):
            module.forward = MethodType(_shared_qwen3_attention_forward, module)
            attention_modules += 1
        elif _is_qwen3_mlp_like(module):
            module.forward = MethodType(_shared_qwen3_mlp_forward, module)
            mlp_modules += 1

    return attention_modules, mlp_modules


def _is_qwen3_attention_like(module: nn.Module) -> bool:
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "head_dim",
        "config",
        "scaling",
    )
    return all(hasattr(module, name) for name in required)


def _is_qwen3_mlp_like(module: nn.Module) -> bool:
    required = ("gate_proj", "up_proj", "down_proj", "act_fn")
    return all(hasattr(module, name) for name in required)


def _uses_fake_quant_linear(module: Any) -> bool:
    return isinstance(module, (FakeQuantLinear, SmoothQuantLinear))


def _shared_prepare_input(modules: tuple[Any, ...], x):
    quant_modules = [module for module in modules if _uses_fake_quant_linear(module)]
    if not quant_modules:
        return x

    smooth_modules = [module for module in quant_modules if isinstance(module, SmoothQuantLinear)]
    if smooth_modules:
        x = smooth_modules[0].smooth_activation(x)
    return fp8_activation_per_token(x)


def _shared_qwen3_attention_forward(
    self: nn.Module,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv_hidden_states = _shared_prepare_input((self.q_proj, self.k_proj, self.v_proj), hidden_states)
    q_input = qkv_hidden_states if _uses_fake_quant_linear(self.q_proj) else hidden_states
    k_input = qkv_hidden_states if _uses_fake_quant_linear(self.k_proj) else hidden_states
    v_input = qkv_hidden_states if _uses_fake_quant_linear(self.v_proj) else hidden_states

    query_states = self.q_norm(self.q_proj(q_input).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(k_input).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(v_input).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = _shared_prepare_input((self.o_proj,), attn_output)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _shared_qwen3_mlp_forward(self: nn.Module, x):
    gate_up_x = _shared_prepare_input((self.gate_proj, self.up_proj), x)
    gate_input = gate_up_x if _uses_fake_quant_linear(self.gate_proj) else x
    up_input = gate_up_x if _uses_fake_quant_linear(self.up_proj) else x

    down_input = self.act_fn(self.gate_proj(gate_input)) * self.up_proj(up_input)
    down_input = _shared_prepare_input((self.down_proj,), down_input)
    return self.down_proj(down_input)
