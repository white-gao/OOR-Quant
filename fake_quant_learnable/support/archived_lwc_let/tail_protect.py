from __future__ import annotations

from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn as nn

from .apply import (
    BaselineQuantSummary,
    _is_qwen3_attention_like,
    _is_qwen3_mlp_like,
    _shared_linear_forward,
    _uses_learnable_quant_linear,
)
from .modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant, ActQuantMode, activation_per_token_qdq_forward, activation_per_token_qdq_ste


def protect_tail_tokens(quantized: torch.Tensor, original: torch.Tensor, tail_tokens: int) -> torch.Tensor:
    """Restore the last tail tokens after activation QDQ.

    For prefill inputs [B, T, H], this protects the final prompt positions. For
    KV-cache decode inputs [B*beam, 1, H], tail_tokens=1 protects the whole
    decode activation. For 2D tensors [T, H], the final rows are protected.
    """
    if tail_tokens <= 0 or original.ndim < 2:
        return quantized
    seq_len = int(original.shape[-2])
    if seq_len <= 0:
        return quantized
    tail = min(int(tail_tokens), seq_len)
    protected = quantized.clone()
    protected[..., -tail:, :] = original[..., -tail:, :]
    return protected


def activation_per_token_qdq_forward_tail_protected(
    x: torch.Tensor,
    *,
    tail_tokens: int,
    qmax: float,
    eps: float,
) -> torch.Tensor:
    quantized = activation_per_token_qdq_forward(x, qmax=qmax, eps=eps)
    return protect_tail_tokens(quantized, x, tail_tokens)


class TailProtectedBaselineFakeQuantLinear(BaselineFakeQuantLinear):
    """W8A8 baseline wrapper that keeps the final activation tokens in FP dtype."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_quant: ActQuant = "none",
        tail_tokens: int = 1,
    ) -> None:
        super().__init__(linear, act_quant=act_quant)
        if tail_tokens < 0:
            raise ValueError(f"tail_tokens must be non-negative, got {tail_tokens}")
        self.tail_tokens = int(tail_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = activation_per_token_qdq_forward_tail_protected(
                x,
                tail_tokens=self.tail_tokens,
                qmax=self.qmax,
                eps=self.eps,
            )
        return self.forward_prepared(x)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", tail_tokens={self.tail_tokens}"


def apply_tail_protected_w8a8(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    act_quant_mode: ActQuantMode = "shared_input",
    tail_tokens: int = 1,
    skip_module_names: Iterable[str] = ("lm_head",),
) -> BaselineQuantSummary:
    """Replace nn.Linear modules with W8A8 wrappers that protect tail tokens."""
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")
    replaced = _replace_children_tail_protected(
        model,
        prefix="",
        act_quant=act_quant,
        tail_tokens=tail_tokens,
        skip_names=set(skip_module_names),
    )
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = install_tail_protected_shared_input_activation_quantization(
            model,
            tail_tokens=tail_tokens,
        )
    return BaselineQuantSummary(
        replaced_linears=replaced,
        skipped_linears=0,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def _replace_children_tail_protected(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    tail_tokens: int,
    skip_names: set[str],
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if child_name in skip_names or full_name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                TailProtectedBaselineFakeQuantLinear(
                    child,
                    act_quant=act_quant,
                    tail_tokens=tail_tokens,
                ),
            )
            replaced += 1
            continue
        replaced += _replace_children_tail_protected(
            child,
            prefix=full_name,
            act_quant=act_quant,
            tail_tokens=tail_tokens,
            skip_names=skip_names,
        )
    return replaced


def install_tail_protected_shared_input_activation_quantization(
    model: nn.Module,
    *,
    tail_tokens: int = 1,
) -> tuple[int, int]:
    """Patch Qwen-style shared-input modules with tail-token protected QDQ."""
    attention_modules = 0
    mlp_modules = 0
    for module in model.modules():
        if _is_qwen3_attention_like(module):
            module._tail_protect_tokens = int(tail_tokens)
            module.forward = MethodType(_tail_protected_qwen3_attention_forward, module)
            attention_modules += 1
        elif _is_qwen3_mlp_like(module):
            module._tail_protect_tokens = int(tail_tokens)
            module.forward = MethodType(_tail_protected_qwen3_mlp_forward, module)
            mlp_modules += 1
    return attention_modules, mlp_modules


def _tail_protected_shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor, *, tail_tokens: int) -> torch.Tensor | None:
    quant_modules = [module for module in modules if _uses_learnable_quant_linear(module)]
    if not quant_modules:
        return None

    first = quant_modules[0]
    if isinstance(first, (LearnableFakeQuantLinear, FrozenLearnedFakeQuantLinear)):
        x = first._let_activation(x)

    if not any(getattr(module, "act_quant", "none") == "per_token" for module in quant_modules):
        return x

    qmax = float(getattr(first, "qmax", 448.0))
    eps = float(getattr(first, "eps", 1e-12))
    if any(isinstance(module, LearnableFakeQuantLinear) for module in quant_modules):
        quantized = activation_per_token_qdq_ste(x, qmax=qmax, eps=eps)
    else:
        quantized = activation_per_token_qdq_forward(x, qmax=qmax, eps=eps)
    return protect_tail_tokens(quantized, x, tail_tokens)


def _tail_tokens_from_module(module: nn.Module) -> int:
    return int(getattr(module, "_tail_protect_tokens", 1))


def _tail_protected_qwen3_attention_forward(
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
    tail_tokens = _tail_tokens_from_module(self)

    qkv_hidden_states = _tail_protected_shared_prepare_input(
        (self.q_proj, self.k_proj, self.v_proj),
        hidden_states,
        tail_tokens=tail_tokens,
    )

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
    o_input = _tail_protected_shared_prepare_input((self.o_proj,), attn_output, tail_tokens=tail_tokens)
    attn_output = _shared_linear_forward(self.o_proj, attn_output, o_input)
    return attn_output, attn_weights


def _tail_protected_qwen3_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    tail_tokens = _tail_tokens_from_module(self)
    gate_up_x = _tail_protected_shared_prepare_input(
        (self.gate_proj, self.up_proj),
        x,
        tail_tokens=tail_tokens,
    )
    gate = _shared_linear_forward(self.gate_proj, x, gate_up_x)
    up = _shared_linear_forward(self.up_proj, x, gate_up_x)
    down_input = self.act_fn(gate) * up
    prepared_down_input = _tail_protected_shared_prepare_input(
        (self.down_proj,),
        down_input,
        tail_tokens=tail_tokens,
    )
    return _shared_linear_forward(self.down_proj, down_input, prepared_down_input)
