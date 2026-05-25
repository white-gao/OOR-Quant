from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator

import torch.nn as nn

from .modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant


@dataclass(frozen=True)
class LearnableLWTSummary:
    replaced_linears: int
    skipped_linears: int


@dataclass(frozen=True)
class BaselineQuantSummary:
    replaced_linears: int
    skipped_linears: int


def apply_baseline_w8a8(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> BaselineQuantSummary:
    """Replace selected nn.Linear modules with min-max FP8 W+A wrappers."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    replaced, skipped = _replace_children_baseline(
        model,
        prefix="",
        act_quant=act_quant,
        skip_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
    )
    return BaselineQuantSummary(replaced_linears=replaced, skipped_linears=skipped)


def apply_learnable_lwt(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
    init_clip_multiplier: float = 1.0,
    min_clip_multiplier: float = 0.05,
    max_clip_multiplier: float = 4.0,
    enable_let: bool = False,
) -> LearnableLWTSummary:
    """Replace selected nn.Linear modules with learnable LWT/LET wrappers."""
    skip_names = set(skip_module_names)
    target_pattern = re.compile(target_regex) if target_regex else None
    skip_pattern = re.compile(skip_regex) if skip_regex else None
    replaced, skipped = _replace_children_learnable(
        model,
        prefix="",
        act_quant=act_quant,
        skip_names=skip_names,
        target_pattern=target_pattern,
        skip_pattern=skip_pattern,
        init_clip_multiplier=init_clip_multiplier,
        min_clip_multiplier=min_clip_multiplier,
        max_clip_multiplier=max_clip_multiplier,
        enable_let=enable_let,
    )
    if enable_let:
        _share_known_let_input_groups(model)
    return LearnableLWTSummary(replaced_linears=replaced, skipped_linears=skipped)


def iter_baseline_w8a8_modules(model: nn.Module) -> Iterator[BaselineFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, BaselineFakeQuantLinear):
            yield module


def iter_learnable_lwt_modules(model: nn.Module) -> Iterator[LearnableFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, LearnableFakeQuantLinear):
            yield module


def learnable_lwt_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    seen: set[int] = set()
    for module in iter_learnable_lwt_modules(model):
        for param in (module.log_clip_multiplier, module.log_let_scale):
            if param is None:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param


def set_learnable_lwt_quant_enabled(model: nn.Module, enabled: bool) -> None:
    for module in iter_learnable_lwt_modules(model):
        module.set_quant_enabled(enabled)


def freeze_learnable_lwt(model: nn.Module) -> int:
    """Replace learnable wrappers with frozen inference wrappers in-place."""
    return _freeze_children(model)


def _replace_children_baseline(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
) -> tuple[int, int]:
    replaced = 0
    skipped = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, (BaselineFakeQuantLinear, LearnableFakeQuantLinear, FrozenLearnedFakeQuantLinear)):
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
            setattr(module, child_name, BaselineFakeQuantLinear(child, act_quant=act_quant))
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children_baseline(
            child,
            prefix=full_name,
            act_quant=act_quant,
            skip_names=skip_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def _replace_children_learnable(
    module: nn.Module,
    *,
    prefix: str,
    act_quant: ActQuant,
    skip_names: set[str],
    target_pattern: re.Pattern[str] | None,
    skip_pattern: re.Pattern[str] | None,
    init_clip_multiplier: float,
    min_clip_multiplier: float,
    max_clip_multiplier: float,
    enable_let: bool,
) -> tuple[int, int]:
    replaced = 0
    skipped = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, (BaselineFakeQuantLinear, LearnableFakeQuantLinear, FrozenLearnedFakeQuantLinear)):
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
                LearnableFakeQuantLinear(
                    child,
                    act_quant=act_quant,
                    init_clip_multiplier=init_clip_multiplier,
                    min_clip_multiplier=min_clip_multiplier,
                    max_clip_multiplier=max_clip_multiplier,
                    enable_let=enable_let,
                ),
            )
            replaced += 1
            continue

        child_replaced, child_skipped = _replace_children_learnable(
            child,
            prefix=full_name,
            act_quant=act_quant,
            skip_names=skip_names,
            target_pattern=target_pattern,
            skip_pattern=skip_pattern,
            init_clip_multiplier=init_clip_multiplier,
            min_clip_multiplier=min_clip_multiplier,
            max_clip_multiplier=max_clip_multiplier,
            enable_let=enable_let,
        )
        replaced += child_replaced
        skipped += child_skipped

    return replaced, skipped


def _share_known_let_input_groups(model: nn.Module) -> None:
    modules = dict(model.named_modules())
    for names in _known_let_input_group_names(modules):
        group = [modules[name] for name in names]
        if not all(isinstance(module, LearnableFakeQuantLinear) for module in group):
            continue
        if not all(module.log_let_scale is not None for module in group):
            continue
        in_features = {module.in_features for module in group}
        if len(in_features) != 1:
            continue
        shared = group[0].log_let_scale
        for module in group[1:]:
            module.log_let_scale = shared


def _known_let_input_group_names(modules: dict[str, nn.Module]) -> Iterator[tuple[str, ...]]:
    for name in sorted(modules):
        if name.endswith(".q_proj"):
            prefix = name[: -len(".q_proj")]
            group = (f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj")
            if all(member in modules for member in group):
                yield group
        elif name.endswith(".gate_proj"):
            prefix = name[: -len(".gate_proj")]
            group = (f"{prefix}.gate_proj", f"{prefix}.up_proj")
            if all(member in modules for member in group):
                yield group


def _freeze_children(module: nn.Module) -> int:
    frozen = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LearnableFakeQuantLinear):
            setattr(module, child_name, child.to_frozen())
            frozen += 1
            continue
        frozen += _freeze_children(child)
    return frozen


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
