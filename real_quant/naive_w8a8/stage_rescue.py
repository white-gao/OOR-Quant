"""Context-local stage routing used only by the SID generation probe.

The normal W8A8 runtime does not set this context, so importing this module
does not alter ordinary quantized inference.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

import torch
import torch.nn as nn


StageName = Literal["a", "b", "c"]
RescueKind = Literal["tail", "all"]

_ACTIVE_STAGES: ContextVar[frozenset[StageName] | None] = ContextVar("real_fp8_probe_active_stages", default=None)
_CURRENT_STAGE: ContextVar[StageName | None] = ContextVar("real_fp8_probe_current_stage", default=None)
_DECODE_STEP: ContextVar[int] = ContextVar("real_fp8_probe_decode_step", default=0)


def _input_seq_len(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if torch.is_tensor(input_ids) and input_ids.ndim >= 2:
        return int(input_ids.shape[-1])
    inputs_embeds = kwargs.get("inputs_embeds")
    if torch.is_tensor(inputs_embeds) and inputs_embeds.ndim >= 3:
        return int(inputs_embeds.shape[-2])
    return None


def install_stage_rescue_model_hook(model: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Track prefill and decode forwards while a probe context is active."""

    def on_model_forward(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if _ACTIVE_STAGES.get() is None:
            return
        seq_len = _input_seq_len(args, kwargs)
        if seq_len is None:
            _CURRENT_STAGE.set(None)
            return
        if seq_len > 1:
            _DECODE_STEP.set(0)
            _CURRENT_STAGE.set("a")
            return
        if seq_len == 1:
            step = _DECODE_STEP.get() + 1
            _DECODE_STEP.set(step)
            _CURRENT_STAGE.set({1: "b", 2: "c"}.get(step))
            return
        _CURRENT_STAGE.set(None)

    return model.register_forward_pre_hook(on_model_forward, with_kwargs=True)


@contextmanager
def stage_rescue_context(stages: set[str] | frozenset[str] | tuple[str, ...]) -> Iterator[None]:
    invalid = set(stages) - {"a", "b", "c"}
    if invalid:
        raise ValueError(f"Unsupported rescue stages: {sorted(invalid)}")
    active = frozenset(stages)
    active_token = _ACTIVE_STAGES.set(active)  # type: ignore[arg-type]
    stage_token = _CURRENT_STAGE.set(None)
    decode_token = _DECODE_STEP.set(0)
    try:
        yield
    finally:
        _DECODE_STEP.reset(decode_token)
        _CURRENT_STAGE.reset(stage_token)
        _ACTIVE_STAGES.reset(active_token)


def rescue_kind_for_input(x: torch.Tensor) -> RescueKind | None:
    """Return the activation rescue path for one ``RealFP8Linear`` call."""

    stages = _ACTIVE_STAGES.get()
    stage = _CURRENT_STAGE.get()
    if not stages or stage not in stages or x.ndim < 3:
        return None
    if stage == "a" and int(x.shape[-2]) > 1:
        return "tail"
    if stage in {"b", "c"} and int(x.shape[-2]) == 1:
        return "all"
    return None


def current_probe_stage() -> StageName | None:
    return _CURRENT_STAGE.get()
