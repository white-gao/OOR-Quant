"""Stage-A weight/activation rescue for the real FP8 runtime."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import apply as real_apply
from .modules import FP8PreparedInput, RealFP8Linear
from .stage_probe_runtime_fixed import activate_stage_activation_rescue
from .stage_rescue import current_probe_stage, install_stage_rescue_model_hook, stage_rescue_context


_MODE: ContextVar[str | None] = ContextVar("real_stage_a_weight_attribution_mode", default=None)


def _stage_a_weight_rescue_active() -> bool:
    return current_probe_stage() == "a" and _MODE.get() in {"w16", "wa16"}


def _fp_weight(module: RealFP8Linear, *, dtype: torch.dtype) -> torch.Tensor:
    weight = getattr(module, "stage_probe_weight_fp", None)
    if weight is None:
        raise RuntimeError("Missing stage_probe_weight_fp on RealFP8Linear.")
    return weight.to(device=module.weight_qdq.device, dtype=dtype)


@contextmanager
def activate_stage_a_weight_attribution(model: nn.Module, mode: str) -> Iterator[None]:
    """Selectively restore Stage-A activations and/or original BF16 weights."""
    if mode not in {"w8a8", "a16", "w16", "wa16"}:
        raise ValueError(f"Unsupported attribution mode {mode!r}")
    if mode == "w8a8":
        yield
        return

    original_prepared = RealFP8Linear.forward_prepared
    original_w8a16 = RealFP8Linear.forward_w8a16
    original_combined = real_apply._combined_w8a16_forward

    def patched_prepared(module: RealFP8Linear, prepared: FP8PreparedInput) -> torch.Tensor:
        output = original_prepared(module, prepared)
        if not _stage_a_weight_rescue_active() or _MODE.get() != "w16" or len(prepared.leading_shape) < 2:
            return output
        qdq_x = (prepared.x_fp8.float() * prepared.scale.float()).reshape(*prepared.leading_shape, module.in_features)
        tail_x = qdq_x[..., -1:, :].to(dtype=module.output_dtype)
        tail = F.linear(tail_x, _fp_weight(module, dtype=module.output_dtype), module._bias_for_output(device=tail_x.device))
        return torch.cat([output[..., :-1, :], tail], dim=-2)

    def patched_w8a16(module: RealFP8Linear, x: torch.Tensor) -> torch.Tensor:
        if not _stage_a_weight_rescue_active() or _MODE.get() != "wa16":
            return original_w8a16(module, x)
        x_2d, leading_shape = module._flatten_input(x)
        x_2d = module._input_for_output_dtype(x_2d)
        bias = module._bias_for_output(device=x_2d.device)
        output = F.linear(x_2d, _fp_weight(module, dtype=module.output_dtype), bias)
        return output.reshape(*leading_shape, module.out_features)

    def patched_combined(modules: tuple[RealFP8Linear, ...], x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not _stage_a_weight_rescue_active() or _MODE.get() != "wa16":
            return original_combined(modules, x)
        if not modules:
            return ()
        output_dtype = modules[0].output_dtype
        if any(module.output_dtype != output_dtype for module in modules):
            raise ValueError("Stage-A combined weight rescue requires matching output dtypes.")
        weight = torch.cat([_fp_weight(module, dtype=output_dtype) for module in modules], dim=0)
        bias_parts = []
        has_bias = False
        for module in modules:
            value = module._bias_for_output(device=x.device)
            if value is None:
                value = torch.zeros(module.out_features, device=x.device, dtype=output_dtype)
            else:
                has_bias = True
            bias_parts.append(value)
        bias = torch.cat(bias_parts, dim=0) if has_bias else None
        output = F.linear(modules[0]._input_for_output_dtype(x), weight, bias)
        return tuple(output.split([module.out_features for module in modules], dim=-1))

    activation_context = (
        activate_stage_activation_rescue(model, {"a"})
        if mode in {"a16", "wa16"}
        else _stage_context_only(model)
    )
    mode_token = _MODE.set(mode)
    RealFP8Linear.forward_prepared = patched_prepared  # type: ignore[method-assign]
    RealFP8Linear.forward_w8a16 = patched_w8a16  # type: ignore[method-assign]
    real_apply._combined_w8a16_forward = patched_combined
    try:
        with activation_context:
            yield
    finally:
        real_apply._combined_w8a16_forward = original_combined
        RealFP8Linear.forward_w8a16 = original_w8a16  # type: ignore[method-assign]
        RealFP8Linear.forward_prepared = original_prepared  # type: ignore[method-assign]
        _MODE.reset(mode_token)


@contextmanager
def _stage_context_only(model: nn.Module) -> Iterator[None]:
    hook = install_stage_rescue_model_hook(model)
    try:
        with stage_rescue_context({"a"}):
            yield
    finally:
        hook.remove()
