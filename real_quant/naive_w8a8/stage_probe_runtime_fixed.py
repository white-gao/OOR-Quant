"""Correct stage-rescue runtime patch for shared-input FP8 linears.

The real runtime fuses Q/K/V and gate/up activation preparation. Therefore a
probe must patch the two routing predicates consulted by both the shared and
ordinary Linear paths, rather than patching ``RealFP8Linear.forward`` alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch.nn as nn

from .modules import RealFP8Linear
from .stage_rescue import install_stage_rescue_model_hook, rescue_kind_for_input, stage_rescue_context


@contextmanager
def activate_stage_activation_rescue(model: nn.Module, stages: set[str]) -> Iterator[None]:
    """Route selected SID prediction stages through the existing W8A16 paths."""

    original_tail_tokens = RealFP8Linear.tail_tokens_for_input
    original_decode_a16 = RealFP8Linear.should_use_decode_a16

    def rescued_tail_tokens(module: RealFP8Linear, x):  # type: ignore[no-untyped-def]
        if rescue_kind_for_input(x) == "tail":
            return 1
        return original_tail_tokens(module, x)

    def rescued_decode_a16(module: RealFP8Linear, x):  # type: ignore[no-untyped-def]
        if rescue_kind_for_input(x) == "all":
            return True
        return original_decode_a16(module, x)

    handle = install_stage_rescue_model_hook(model)
    RealFP8Linear.tail_tokens_for_input = rescued_tail_tokens  # type: ignore[method-assign]
    RealFP8Linear.should_use_decode_a16 = rescued_decode_a16  # type: ignore[method-assign]
    try:
        with stage_rescue_context(stages):
            yield
    finally:
        RealFP8Linear.tail_tokens_for_input = original_tail_tokens  # type: ignore[method-assign]
        RealFP8Linear.should_use_decode_a16 = original_decode_a16  # type: ignore[method-assign]
        handle.remove()
