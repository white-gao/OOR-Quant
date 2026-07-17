"""Temporary runtime patch for the stage-rescue probe.

This module deliberately avoids changing the normal ``RealFP8Linear`` source
path. The class method is patched only inside the context manager below and is
restored before the probe moves to its next variant.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch.nn as nn

from .modules import RealFP8Linear
from .stage_rescue import install_stage_rescue_model_hook, rescue_kind_for_input, stage_rescue_context


@contextmanager
def activate_stage_activation_rescue(model: nn.Module, stages: set[str]) -> Iterator[None]:
    """Use W8A16 only at the requested SID prediction stages.

    Stage A protects one tail token in prefill. Stages B/C protect the complete
    single-token decode input. Quantized weights are unchanged.
    """

    original_forward = RealFP8Linear.forward

    def rescued_forward(module: RealFP8Linear, x):  # type: ignore[no-untyped-def]
        kind = rescue_kind_for_input(x)
        if kind == "tail":
            return module.forward_tail_protected(x, tail_tokens=1)
        if kind == "all":
            return module.forward_w8a16(x)
        return original_forward(module, x)

    handle = install_stage_rescue_model_hook(model)
    RealFP8Linear.forward = rescued_forward  # type: ignore[method-assign]
    try:
        with stage_rescue_context(stages):
            yield
    finally:
        RealFP8Linear.forward = original_forward  # type: ignore[method-assign]
        handle.remove()
