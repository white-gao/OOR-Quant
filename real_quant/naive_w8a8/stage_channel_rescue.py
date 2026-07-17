"""Selective Stage-A activation-channel rescue for the real FP8 runtime."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FP8PreparedInput, RealFP8Linear
from .stage_rescue import current_probe_stage, install_stage_rescue_model_hook, stage_rescue_context


_PREPARED_TAILS: ContextVar[dict[int, torch.Tensor] | None] = ContextVar("stage_channel_rescue_tails", default=None)


def _label(module: RealFP8Linear) -> str:
    name = getattr(module, "stage_channel_rescue_name", "")
    if any(f".self_attn.{projection}" in name for projection in ("q_proj", "k_proj", "v_proj")):
        suffix = "qkv_input"
    elif any(f".mlp.{projection}" in name for projection in ("gate_proj", "up_proj")):
        suffix = "gate_up_input"
    elif ".self_attn.o_proj" in name:
        suffix = "o_proj"
    elif ".mlp.down_proj" in name:
        suffix = "down_proj"
    else:
        suffix = name.rsplit(".", 1)[-1]
    layer = name.split(".layers.")[1].split(".")[0]
    return f"layer{layer}.{suffix}"


@contextmanager
def activate_stage_a_channel_rescue(
    model: nn.Module,
    channel_indices: Mapping[str, torch.Tensor],
) -> Iterator[None]:
    """Restore selected Stage-A activation features with a BF16 correction.

    For selected feature set C this computes the exact correction
    ``(x_C - xhat_C) Wq_C^T`` after the regular FP8 GEMM.  Thus weights remain
    quantized and all non-selected activation channels remain FP8-QDQ.
    """
    original_prepare = RealFP8Linear.prepare_input
    original_forward_prepared = RealFP8Linear.forward_prepared
    modules = list(model.named_modules())
    for name, module in modules:
        if isinstance(module, RealFP8Linear):
            module.stage_channel_rescue_name = name

    def clear_tails(*_args, **_kwargs) -> None:
        tails = _PREPARED_TAILS.get()
        if tails is not None:
            tails.clear()

    def patched_prepare(module: RealFP8Linear, x: torch.Tensor) -> FP8PreparedInput:
        prepared = original_prepare(module, x)
        if current_probe_stage() == "a" and x.ndim >= 3 and int(x.shape[-2]) > 1:
            tails = _PREPARED_TAILS.get()
            if tails is not None:
                tails[id(prepared)] = x.detach().float().reshape(-1, module.in_features)[-1:]
        return prepared

    def patched_forward_prepared(module: RealFP8Linear, prepared: FP8PreparedInput) -> torch.Tensor:
        output = original_forward_prepared(module, prepared)
        if current_probe_stage() != "a" or len(prepared.leading_shape) < 2:
            return output
        indices = channel_indices.get(_label(module))
        tails = _PREPARED_TAILS.get()
        source = None if tails is None else tails.get(id(prepared))
        if indices is None or source is None or indices.numel() == 0:
            return output
        index = indices.to(device=source.device, dtype=torch.long)
        qdq = (prepared.x_fp8.float() * prepared.scale.float())[-1:, index]
        delta = source[:, index] - qdq
        correction = F.linear(
            delta.to(dtype=module.output_dtype),
            module.weight_qdq[:, index],
            bias=None,
        )
        correction_tail = correction.reshape(*([1] * (output.ndim - 1)), module.out_features).expand_as(output[..., -1:, :])
        corrected_tail = output[..., -1:, :] + correction_tail
        return torch.cat([output[..., :-1, :], corrected_tail], dim=-2)

    stage_hook = install_stage_rescue_model_hook(model)
    cleanup_hook = model.register_forward_hook(clear_tails)
    token = _PREPARED_TAILS.set({})
    RealFP8Linear.prepare_input = patched_prepare  # type: ignore[method-assign]
    RealFP8Linear.forward_prepared = patched_forward_prepared  # type: ignore[method-assign]
    try:
        with stage_rescue_context({"a"}):
            yield
    finally:
        RealFP8Linear.forward_prepared = original_forward_prepared  # type: ignore[method-assign]
        RealFP8Linear.prepare_input = original_prepare  # type: ignore[method-assign]
        cleanup_hook.remove()
        stage_hook.remove()
        _PREPARED_TAILS.reset(token)
        for _name, module in modules:
            if isinstance(module, RealFP8Linear) and hasattr(module, "stage_channel_rescue_name"):
                delattr(module, "stage_channel_rescue_name")
