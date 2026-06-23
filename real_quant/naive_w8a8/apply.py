from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import re
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn

from .modules import FP8_MAX, FP8PreparedInput, RealFP8Linear


@dataclass(frozen=True)
class NaiveW8A8Summary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def apply_naive_w8a8(
    model: nn.Module,
    *,
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
    qmax: float = FP8_MAX,
    eps: float = 1e-12,
    output_dtype: torch.dtype | None = torch.bfloat16,
    use_fast_accum: bool = False,
) -> NaiveW8A8Summary:
    """Replace selected nn.Linear modules with real FP8 W8A8 wrappers."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    replaced, skipped = _replace_children(
        model,
        prefix="",
        skip_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
        qmax=qmax,
        eps=eps,
        output_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )
    shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(model)
    return NaiveW8A8Summary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def iter_real_fp8_linears(model: nn.Module) -> Iterator[RealFP8Linear]:
    for module in model.modules():
        if isinstance(module, RealFP8Linear):
            yield module


def install_shared_input_activation_quantization(model: nn.Module) -> tuple[int, int]:
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


def _replace_children(
    module: nn.Module,
    *,
    prefix: str,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
    qmax: float,
    eps: float,
    output_dtype: torch.dtype | None,
    use_fast_accum: bool,
) -> tuple[int, int]:
    replaced = 0
    skipped = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, RealFP8Linear):
            continue
        if isinstance(child, nn.Linear):
            if _should_skip(
                child_name=child_name,
                full_name=full_name,
                skip_names=skip_names,
                target_pattern=target_pattern,
                skip_pattern=skip_pattern,
            ):
                skipped += 1
                continue
            setattr(
                module,
                child_name,
                RealFP8Linear.from_linear(
                    child,
                    qmax=qmax,
                    eps=eps,
                    output_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                ),
            )
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children(
            child,
            prefix=full_name,
            skip_names=skip_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
            qmax=qmax,
            eps=eps,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        replaced += child_replaced
        skipped += child_skipped
    return replaced, skipped


def _should_skip(
    *,
    child_name: str,
    full_name: str,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> bool:
    if child_name in skip_names or full_name in skip_names:
        return True
    if skip_pattern is not None and skip_pattern.search(full_name) is not None:
        return True
    if target_pattern is not None and target_pattern.search(full_name) is None:
        return True
    return False


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
        "attention_dropout",
        "layer_idx",
        "sliding_window",
        "scaling",
    )
    return all(hasattr(module, name) for name in required)


def _is_qwen3_mlp_like(module: nn.Module) -> bool:
    required = ("gate_proj", "up_proj", "down_proj", "act_fn")
    return all(hasattr(module, name) for name in required)


def _uses_real_fp8_linear(module: Any) -> bool:
    return isinstance(module, RealFP8Linear)


def _shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor) -> FP8PreparedInput | None:
    quant_modules = [module for module in modules if _uses_real_fp8_linear(module)]
    if not quant_modules:
        return None
    return quant_modules[0].prepare_input(x)


def _shared_linear_forward(module: Any, raw_x: torch.Tensor, prepared_x: FP8PreparedInput | None) -> torch.Tensor:
    if prepared_x is not None and _uses_real_fp8_linear(module):
        return module.forward_prepared(prepared_x)
    return module(raw_x)


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
    o_input = _shared_prepare_input((self.o_proj,), attn_output)
    attn_output = _shared_linear_forward(self.o_proj, attn_output, o_input)
    return attn_output, attn_weights


def _shared_qwen3_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate_up_x = _shared_prepare_input((self.gate_proj, self.up_proj), x)
    gate = _shared_linear_forward(self.gate_proj, x, gate_up_x)
    up = _shared_linear_forward(self.up_proj, x, gate_up_x)
    down_input = self.act_fn(gate) * up
    prepared_down_input = _shared_prepare_input((self.down_proj,), down_input)
    return _shared_linear_forward(self.down_proj, down_input, prepared_down_input)
