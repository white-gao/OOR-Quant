from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import re
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FP8_MAX, ActivationQuantMode, FP8PreparedInput, FP8TailPreparedInput, RealFP8Linear


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
    activation_quant_mode: ActivationQuantMode = "dynamic",
    decode_a16_when_single_token: bool = False,
    activation_tail_tokens: int = 0,
    gptq_hessians: Mapping[str, torch.Tensor] | None = None,
    gptq_damp_percent: float = 0.01,
    gptq_block_size: int = 128,
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
        activation_quant_mode=activation_quant_mode,
        decode_a16_when_single_token=decode_a16_when_single_token,
        activation_tail_tokens=activation_tail_tokens,
        gptq_hessians=gptq_hessians,
        gptq_damp_percent=gptq_damp_percent,
        gptq_block_size=gptq_block_size,
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
    activation_quant_mode: ActivationQuantMode,
    decode_a16_when_single_token: bool,
    activation_tail_tokens: int,
    gptq_hessians: Mapping[str, torch.Tensor] | None,
    gptq_damp_percent: float,
    gptq_block_size: int,
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
            if gptq_hessians is None:
                quant_child = RealFP8Linear.from_linear(
                    child,
                    qmax=qmax,
                    eps=eps,
                    output_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    activation_tail_tokens=activation_tail_tokens,
                )
            else:
                hessian = gptq_hessians.get(full_name)
                if hessian is None:
                    raise KeyError(f"Missing GPTQ Hessian for Linear module: {full_name}")
                quant_child = RealFP8Linear.from_gptq_linear(
                    child,
                    hessian,
                    qmax=qmax,
                    eps=eps,
                    output_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                    activation_quant_mode=activation_quant_mode,
                    decode_a16_when_single_token=decode_a16_when_single_token,
                    activation_tail_tokens=activation_tail_tokens,
                    damp_percent=gptq_damp_percent,
                    block_size=gptq_block_size,
                )
            setattr(module, child_name, quant_child)
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
            activation_quant_mode=activation_quant_mode,
            decode_a16_when_single_token=decode_a16_when_single_token,
            activation_tail_tokens=activation_tail_tokens,
            gptq_hessians=gptq_hessians,
            gptq_damp_percent=gptq_damp_percent,
            gptq_block_size=gptq_block_size,
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


def _shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor) -> FP8PreparedInput | FP8TailPreparedInput | None:
    quant_modules = [module for module in modules if _uses_real_fp8_linear(module)]
    if not quant_modules:
        return None
    first = quant_modules[0]
    tail_tokens = first.tail_tokens_for_input(x)
    if tail_tokens > 0:
        seq_len = int(x.shape[-2])
        if tail_tokens >= seq_len:
            return None
        main_x = x[..., :-tail_tokens, :]
        tail_x = x[..., -tail_tokens:, :]
        return FP8TailPreparedInput(main=first.prepare_input(main_x), tail_x=tail_x)
    if first.should_use_decode_a16(x):
        return None
    return first.prepare_input(x)


def _shared_linear_forward(
    module: Any,
    raw_x: torch.Tensor,
    prepared_x: FP8PreparedInput | FP8TailPreparedInput | None,
) -> torch.Tensor:
    if prepared_x is not None and _uses_real_fp8_linear(module):
        if isinstance(prepared_x, FP8TailPreparedInput):
            y_main = module.forward_prepared(prepared_x.main)
            y_tail = module.forward_w8a16(prepared_x.tail_x)
            return torch.cat([y_main, y_tail], dim=-2)
        return module.forward_prepared(prepared_x)
    return module(raw_x)


def _combined_w8a16_forward(modules: tuple[RealFP8Linear, ...], x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not modules:
        return ()
    output_dtype = modules[0].output_dtype
    for module in modules:
        if not isinstance(module, RealFP8Linear):
            raise TypeError(f"Expected RealFP8Linear, got {type(module)!r}.")
        if module.output_dtype != output_dtype:
            raise ValueError("Combined W8A16 tail forward requires matching output dtypes.")

    weight = torch.cat([module.weight_qdq for module in modules], dim=0)
    if all(module.bias is None for module in modules):
        bias = None
    else:
        bias_parts = []
        for module in modules:
            module_bias = module._bias_for_output(device=x.device)
            if module_bias is None:
                module_bias = torch.zeros(module.out_features, device=x.device, dtype=output_dtype)
            bias_parts.append(module_bias)
        bias = torch.cat(bias_parts, dim=0)

    x_for_linear = modules[0]._input_for_output_dtype(x)
    combined = F.linear(x_for_linear, weight, bias)
    return tuple(combined.split([module.out_features for module in modules], dim=-1))


def _tail_combined_outputs(
    modules: tuple[Any, ...],
    prepared_x: FP8TailPreparedInput,
) -> tuple[torch.Tensor, ...]:
    quant_modules = tuple(module for module in modules if _uses_real_fp8_linear(module))
    if len(quant_modules) != len(modules):
        raise TypeError("Tail-combined output requires all modules to be RealFP8Linear.")
    main_outputs = tuple(module.forward_prepared(prepared_x.main) for module in quant_modules)
    tail_outputs = _combined_w8a16_forward(quant_modules, prepared_x.tail_x)
    return tuple(torch.cat([main, tail], dim=-2) for main, tail in zip(main_outputs, tail_outputs))


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

    if isinstance(qkv_hidden_states, FP8TailPreparedInput):
        query_raw, key_raw, value_raw = _tail_combined_outputs(
            (self.q_proj, self.k_proj, self.v_proj),
            qkv_hidden_states,
        )
    else:
        query_raw = _shared_linear_forward(self.q_proj, hidden_states, qkv_hidden_states)
        key_raw = _shared_linear_forward(self.k_proj, hidden_states, qkv_hidden_states)
        value_raw = _shared_linear_forward(self.v_proj, hidden_states, qkv_hidden_states)

    query_states = self.q_norm(query_raw.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_raw.view(hidden_shape)).transpose(1, 2)
    value_states = value_raw.view(hidden_shape).transpose(1, 2)

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
    if isinstance(gate_up_x, FP8TailPreparedInput):
        gate, up = _tail_combined_outputs((self.gate_proj, self.up_proj), gate_up_x)
    else:
        gate = _shared_linear_forward(self.gate_proj, x, gate_up_x)
        up = _shared_linear_forward(self.up_proj, x, gate_up_x)
    down_input = self.act_fn(gate) * up
    prepared_down_input = _shared_prepare_input((self.down_proj,), down_input)
    return _shared_linear_forward(self.down_proj, down_input, prepared_down_input)
