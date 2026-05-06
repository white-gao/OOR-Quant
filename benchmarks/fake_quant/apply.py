from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Tuple

import torch.nn as nn

from .modules import FakeQuantLinear
from .quant import ActQuant


@dataclass
class FakeQuantSummary:
    replaced_linears: int
    skipped_linears: int


def apply_fp8_fake_quant(
    model: nn.Module,
    *,
    act_quant: ActQuant = "none",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
) -> FakeQuantSummary:
    """Replace Linear modules with FP8 fake-quant Linear wrappers."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    replaced, skipped = _replace_children(
        module=model,
        prefix="",
        act_quant=act_quant,
        skip_module_names=skip_names,
        target_pattern=target_pattern,
    )
    return FakeQuantSummary(replaced_linears=replaced, skipped_linears=skipped)


def _replace_children(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    skip_module_names: set[str],
    target_pattern: re.Pattern[str] | None,
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
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped
