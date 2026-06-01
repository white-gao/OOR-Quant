from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
import re
from typing import Any, Iterable, Iterator, Literal, Mapping

import torch
import torch.nn as nn

from .modules import BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear, LearnableFakeQuantLinear
from .quant import ActQuant, ActQuantMode, activation_per_token_qdq_forward, activation_per_token_qdq_ste


SmoothScope = Literal["all", "omni"]
OMNI_SMOOTH_LINEAR_LEAF_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


@dataclass(frozen=True)
class LearnableLWTSummary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


@dataclass(frozen=True)
class BaselineQuantSummary:
    replaced_linears: int
    skipped_linears: int
    shared_attention_modules: int = 0
    shared_mlp_modules: int = 0


def apply_baseline_w8a8(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    act_quant_mode: ActQuantMode = "per_linear",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
) -> BaselineQuantSummary:
    """Replace selected nn.Linear modules with min-max FP8 W+A wrappers."""
    _validate_act_quant_mode(act_quant=act_quant, act_quant_mode=act_quant_mode)
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
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(model)
    return BaselineQuantSummary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def apply_learnable_lwt(
    model: nn.Module,
    *,
    act_quant: ActQuant = "per_token",
    act_quant_mode: ActQuantMode = "per_linear",
    skip_module_names: Iterable[str] = ("lm_head",),
    target_regex: str | None = None,
    skip_regex: str | None = None,
    init_clip_multiplier: float = 1.0,
    min_clip_multiplier: float = 0.05,
    max_clip_multiplier: float = 4.0,
    enable_let: bool = False,
    let_scope: SmoothScope = "all",
) -> LearnableLWTSummary:
    """Replace selected nn.Linear modules with learnable LWT/LET wrappers."""
    _validate_act_quant_mode(act_quant=act_quant, act_quant_mode=act_quant_mode)
    _validate_smooth_scope(let_scope)
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
        let_scope=let_scope,
    )
    if enable_let:
        _share_known_let_input_groups(model)
    shared_attention_modules = 0
    shared_mlp_modules = 0
    if act_quant == "per_token" and act_quant_mode == "shared_input":
        shared_attention_modules, shared_mlp_modules = install_shared_input_activation_quantization(model)
    return LearnableLWTSummary(
        replaced_linears=replaced,
        skipped_linears=skipped,
        shared_attention_modules=shared_attention_modules,
        shared_mlp_modules=shared_mlp_modules,
    )


def iter_baseline_w8a8_modules(model: nn.Module) -> Iterator[BaselineFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, BaselineFakeQuantLinear):
            yield module


def iter_learnable_lwt_modules(model: nn.Module) -> Iterator[LearnableFakeQuantLinear]:
    for module in model.modules():
        if isinstance(module, LearnableFakeQuantLinear):
            yield module


def learnable_lwt_parameters(
    model: nn.Module,
    *,
    include_lwt: bool = True,
    include_let: bool = True,
) -> Iterator[nn.Parameter]:
    seen: set[int] = set()
    for module in iter_learnable_lwt_modules(model):
        params = (
            module.log_clip_multiplier if include_lwt else None,
            module.log_let_scale if include_let else None,
        )
        for param in params:
            if param is None:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param



def install_shared_input_activation_quantization(model: nn.Module) -> tuple[int, int]:
    """Patch Qwen-style attention/MLP modules to reuse activation QDQ at shared inputs."""
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


def _validate_act_quant_mode(*, act_quant: ActQuant, act_quant_mode: ActQuantMode) -> None:
    if act_quant_mode not in ("per_linear", "shared_input"):
        raise ValueError(f"Unsupported act_quant_mode: {act_quant_mode}")
    if act_quant == "none" and act_quant_mode != "per_linear":
        raise ValueError("act_quant_mode is only meaningful when activation quantization is enabled.")


def _validate_smooth_scope(scope: str) -> None:
    if scope not in ("all", "omni"):
        raise ValueError(f"Unsupported smooth scope: {scope!r}")


def should_apply_smooth_transform(full_name: str, scope: SmoothScope) -> bool:
    """Return whether LET/SmoothQuant should use an equivalent-transform scale."""
    _validate_smooth_scope(scope)
    if scope == "all":
        return True
    if full_name == "":
        return True
    return full_name.rsplit(".", 1)[-1] in OMNI_SMOOTH_LINEAR_LEAF_NAMES


def set_learnable_lwt_quant_enabled(model: nn.Module, enabled: bool) -> None:
    for module in iter_learnable_lwt_modules(model):
        module.set_quant_enabled(enabled)


def export_learned_quant_params(model: nn.Module) -> dict[str, Any]:
    """Export lightweight learned LWT/LET parameters from learnable wrappers."""
    modules: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LearnableFakeQuantLinear):
            continue
        modules[name] = {
            "act_quant": module.act_quant,
            "enable_let": module.let_scale is not None,
            "qmax": module.qmax,
            "eps": module.eps,
            "min_clip_multiplier": module.min_clip_multiplier,
            "max_clip_multiplier": module.max_clip_multiplier,
            "min_let_scale": module.min_let_scale,
            "max_let_scale": module.max_let_scale,
            "clip_multiplier": module.clip_multiplier.detach().cpu(),
            "let_scale": None if module.let_scale is None else module.let_scale.detach().cpu(),
        }
    return {"format_version": 1, "modules": modules}


def apply_learned_quant_params(model: nn.Module, params: Mapping[str, Any]) -> int:
    """Apply exported LWT/LET params to matching Linear modules and freeze them."""
    modules = params.get("modules", params)
    if not isinstance(modules, Mapping):
        raise TypeError("Expected learned quant params to contain a modules mapping.")

    replaced = 0
    for name, entry in modules.items():
        if name == "":
            raise ValueError("Root-module learned params must be applied by the caller.")
        parent, child_name, child = _resolve_child_module(model, str(name))
        setattr(parent, child_name, _build_frozen_learned_linear(child, entry))
        replaced += 1
    return replaced


def learned_quantized_module_from_params(module: nn.Module, params: Mapping[str, Any]) -> tuple[nn.Module, int]:
    """Return a module with exported learned params applied, supporting root Linear layers."""
    modules = params.get("modules", params)
    if not isinstance(modules, Mapping):
        raise TypeError("Expected learned quant params to contain a modules mapping.")
    if set(modules.keys()) == {""}:
        return _build_frozen_learned_linear(module, modules[""]), 1
    return module, apply_learned_quant_params(module, params)


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
            setattr(module, child_name, BaselineFakeQuantLinear(child, act_quant=act_quant)) # 将nn.linear替换为BaselineFakeQuantLinear
            replaced += 1
            continue
        # 这里迭代所有子模块，进行替换，例如attn -> attn.q_proj, attn.k_proj, attn.v_proj
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
    let_scope: SmoothScope,
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
                    enable_let=enable_let and should_apply_smooth_transform(full_name, let_scope),
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
            let_scope=let_scope,
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


def _resolve_child_module(model: nn.Module, name: str) -> tuple[nn.Module, str, nn.Module]:
    parts = name.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    child_name = parts[-1]
    if not hasattr(parent, child_name):
        raise AttributeError(f"Could not find child module {name!r}.")
    child = getattr(parent, child_name)
    if not isinstance(child, nn.Module):
        raise TypeError(f"Expected {name!r} to resolve to nn.Module, got {type(child)!r}.")
    return parent, child_name, child


def _build_frozen_learned_linear(module: nn.Module, entry: Mapping[str, Any]) -> FrozenLearnedFakeQuantLinear:
    if isinstance(module, LearnableFakeQuantLinear):
        wrapper = module
    elif isinstance(module, nn.Linear):
        wrapper = LearnableFakeQuantLinear(
            module,
            act_quant=entry.get("act_quant", "per_token"),
            init_clip_multiplier=1.0,
            min_clip_multiplier=float(entry.get("min_clip_multiplier", 0.05)),
            max_clip_multiplier=float(entry.get("max_clip_multiplier", 4.0)),
            enable_let=bool(entry.get("enable_let", entry.get("let_scale") is not None)),
            init_let_scale=1.0,
            min_let_scale=float(entry.get("min_let_scale", 0.05)),
            max_let_scale=float(entry.get("max_let_scale", 20.0)),
            qmax=float(entry.get("qmax", 448.0)),
            eps=float(entry.get("eps", 1e-12)),
        )
    else:
        raise TypeError(f"Expected nn.Linear or LearnableFakeQuantLinear, got {type(module)!r}.")

    _load_entry_into_learnable_linear(wrapper, entry)
    return wrapper.to_frozen()


def _load_entry_into_learnable_linear(module: LearnableFakeQuantLinear, entry: Mapping[str, Any]) -> None:
    clip_multiplier = torch.as_tensor(
        entry["clip_multiplier"],
        device=module.log_clip_multiplier.device,
        dtype=module.log_clip_multiplier.dtype,
    )
    if tuple(clip_multiplier.shape) != tuple(module.log_clip_multiplier.shape):
        raise ValueError(
            f"clip_multiplier shape mismatch: expected {tuple(module.log_clip_multiplier.shape)}, "
            f"got {tuple(clip_multiplier.shape)}"
        )
    with torch.no_grad():
        module.log_clip_multiplier.copy_(torch.log(clip_multiplier))

    let_scale = entry.get("let_scale")
    if let_scale is None:
        if module.log_let_scale is not None:
            raise ValueError("Loaded params do not contain LET scale but target module has LET enabled.")
        return
    if module.log_let_scale is None:
        raise ValueError("Loaded params contain LET scale but target module has LET disabled.")
    let_tensor = torch.as_tensor(
        let_scale,
        device=module.log_let_scale.device,
        dtype=module.log_let_scale.dtype,
    ).reshape_as(module.log_let_scale)
    with torch.no_grad():
        module.log_let_scale.copy_(torch.log(let_tensor))


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


def _uses_learnable_quant_linear(module: Any) -> bool:
    if isinstance(module, LearnableFakeQuantLinear):
        return module.quant_enabled
    return isinstance(module, (BaselineFakeQuantLinear, FrozenLearnedFakeQuantLinear))


def _shared_prepare_input(modules: tuple[Any, ...], x: torch.Tensor) -> torch.Tensor | None:
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
        return activation_per_token_qdq_ste(x, qmax=qmax, eps=eps)
    return activation_per_token_qdq_forward(x, qmax=qmax, eps=eps)


def _shared_linear_forward(module: Any, raw_x: torch.Tensor, prepared_x: torch.Tensor | None) -> torch.Tensor:
    if prepared_x is not None and _uses_learnable_quant_linear(module):
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
            key_states, value_states, self.layer_idx, cache_kwargs
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
