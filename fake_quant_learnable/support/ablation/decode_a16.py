from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn as nn

from fake_quant_learnable.apply import (
    BaselineQuantSummary,
    _is_qwen3_attention_like,
    _is_qwen3_mlp_like,
    _shared_linear_forward,
)
from fake_quant_learnable.modules import BaselineFakeQuantLinear
from fake_quant_learnable.quant import ActQuant, ActQuantMode, activation_per_token_qdq_forward


def protect_decode_step_activation(quantized: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Bypass activation QDQ only for single-token decode inputs."""
    if original.ndim >= 2 and int(original.shape[-2]) == 1:
        return original
    return quantized


def activation_per_token_qdq_forward_decode_a16(
    x: torch.Tensor,
    *,
    qmax: float,
    eps: float,
) -> torch.Tensor:
    quantized = activation_per_token_qdq_forward(x, qmax=qmax, eps=eps)
    return protect_decode_step_activation(quantized, x)


class DecodeA16BaselineFakeQuantLinear(BaselineFakeQuantLinear):
    """W8A8 wrapper that keeps only single-token decode activations in FP dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward_decode_a16(
                x,
                qmax=self.qmax,
                eps=self.eps,
            )
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return super().extra_repr() + ", decode_a16=True"


def apply_decode_a16_w8a8(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    act_quant_mode: ActQuantMode = "shared_input",
    skip_module_names: Iterable[str] = ("lm_head",),
) -> BaselineQuantSummary:
    """Replace nn.Linear modules with W8A8 wrappers that bypass A8 only at decode steps."""
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")

    replaced = _replace_children_decode_a16(
        model,
        prefix="",
        act_quant=act_quant,
        skip_names=set(skip_module_names),
    )
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = install_decode_a16_shared_input_activation_quantization(model)
    return BaselineQuantSummary(
        replaced_linears=replaced,
        skipped_linears=0,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def _replace_children_decode_a16(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    skip_names: set[str],
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if child_name in skip_names or full_name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, DecodeA16BaselineFakeQuantLinear(child, act_quant=act_quant))
            replaced += 1
            continue
        replaced += _replace_children_decode_a16(
            child,
            prefix=full_name,
            act_quant=act_quant,
            skip_names=skip_names,
        )
    return replaced


def install_decode_a16_shared_input_activation_quantization(model: nn.Module) -> tuple[int, int]:
    """Patch Qwen-style shared-input modules with decode-only A16 activation QDQ."""
    attention_modules = 0
    mlp_modules = 0
    for module in model.modules():
        if _is_qwen3_attention_like(module):
            module.forward = MethodType(_decode_a16_qwen3_attention_forward, module)
            attention_modules += 1
        elif _is_qwen3_mlp_like(module):
            module.forward = MethodType(_decode_a16_qwen3_mlp_forward, module)
            mlp_modules += 1
    return attention_modules, mlp_modules


def _uses_decode_a16_linear(module: Any) -> bool:
    return isinstance(module, DecodeA16BaselineFakeQuantLinear)


def _decode_a16_shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor) -> torch.Tensor | None:
    quant_modules = [module for module in modules if _uses_decode_a16_linear(module)]
    if not quant_modules:
        return None
    first = quant_modules[0]
    if not any(getattr(module, "act_quant", "none") == "per_token" for module in quant_modules):
        return x
    qmax = float(getattr(first, "qmax", 448.0))
    eps = float(getattr(first, "eps", 1e-12))
    return activation_per_token_qdq_forward_decode_a16(x, qmax=qmax, eps=eps)


def _decode_a16_qwen3_attention_forward(
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

    qkv_hidden_states = _decode_a16_shared_prepare_input((self.q_proj, self.k_proj, self.v_proj), hidden_states)

    query_states = self.q_norm(
        _shared_linear_forward(self.q_proj, hidden_states, qkv_hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = self.k_norm(
        _shared_linear_forward(self.k_proj, hidden_states, qkv_hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = _shared_linear_forward(self.v_proj, hidden_states, qkv_hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
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
    o_input = _decode_a16_shared_prepare_input((self.o_proj,), attn_output)
    attn_output = _shared_linear_forward(self.o_proj, attn_output, o_input)
    return attn_output, attn_weights


def _decode_a16_qwen3_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_up_x = _decode_a16_shared_prepare_input((self.gate_proj, self.up_proj), x)
    gate = _shared_linear_forward(self.gate_proj, x, gate_up_x)
    up = _shared_linear_forward(self.up_proj, x, gate_up_x)
    down_input = self.act_fn(gate) * up
    prepared_down_input = _decode_a16_shared_prepare_input((self.down_proj,), down_input)
    return _shared_linear_forward(self.down_proj, down_input, prepared_down_input)

