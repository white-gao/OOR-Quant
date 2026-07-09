from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

import torch


SID_ITEM_RE = re.compile(
    r"<\|sid_begin\|>"
    r"<s_a_[^>]+><s_b_[^>]+><s_c_[^>]+>"
    r"<\|sid_end\|>"
)
SID_TOKEN_RE = re.compile(r"<\|sid_begin\|>|<\|sid_end\|>|<s_[abc]_[^>]+>")
SID_BOUNDARY_RE = re.compile(r"<\|sid_begin\|>|<\|sid_end\|>")
SID_CODE_RE = re.compile(r"<s_[abc]_[^>]+>")
SECTION_BREAK_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]")

PROMPT_TOKEN_GROUPS = ("text", "history_sid", "interest_sid", "sid_boundary")
PROMPT_TOKEN_GROUP_IDS = {group: idx for idx, group in enumerate(PROMPT_TOKEN_GROUPS)}
SLOT_TOKEN_GROUPS = ("text", "sid_a", "sid_b", "sid_c", "boundary")
SLOT_TOKEN_GROUP_IDS = {group: idx for idx, group in enumerate(SLOT_TOKEN_GROUPS)}


@dataclass(frozen=True)
class PromptTokenWeightConfig:
    history_sid_weight: float = 1.0
    interest_sid_weight: float = 5.0
    text_weight: float = 10.0
    sid_boundary_weight: float = 2.0
    normalize_mean: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROMPT_TOKEN_WEIGHT_CONFIG = PromptTokenWeightConfig()


@dataclass(frozen=True)
class SlotTokenWeightConfig:
    text_weight: float = 10.0
    sid_a_weight: float = 5.0
    sid_b_weight: float = 2.0
    sid_c_weight: float = 2.0
    boundary_weight: float = 2.0
    normalize_mean: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SLOT_TOKEN_WEIGHT_CONFIG = SlotTokenWeightConfig()


def build_prompt_token_weight_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device | str,
    config: PromptTokenWeightConfig = DEFAULT_PROMPT_TOKEN_WEIGHT_CONFIG,
) -> list[torch.Tensor]:
    """Build per-token calibration weights from prompt role spans.

    The weights are used only while collecting GPTQ Hessians. They do not affect
    generation-time tokenization or model forward behavior.
    """
    return [
        build_prompt_token_weights(tokenizer=tokenizer, prompt=prompt, device=device, config=config)
        for prompt in prompts
    ]


def build_prompt_slot_token_weight_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device | str,
    config: SlotTokenWeightConfig = DEFAULT_SLOT_TOKEN_WEIGHT_CONFIG,
) -> list[torch.Tensor]:
    """Build calibration weights from SID slot spans: text/sid_a/sid_b/sid_c/boundary."""
    return [
        build_prompt_slot_token_weights(tokenizer=tokenizer, prompt=prompt, device=device, config=config)
        for prompt in prompts
    ]


def build_prompt_token_weights(
    *,
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
    config: PromptTokenWeightConfig = DEFAULT_PROMPT_TOKEN_WEIGHT_CONFIG,
) -> torch.Tensor:
    encoded, offsets = _tokenize_with_offsets(tokenizer, prompt)
    input_ids = _as_tensor(encoded["input_ids"])
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected single-prompt input_ids shape [1, seq], got {tuple(input_ids.shape)}")

    if offsets is None:
        weights = _fallback_weights_from_decoded_tokens(tokenizer, input_ids, config=config)
    else:
        weights = _weights_from_offsets(prompt, offsets, config=config)

    if weights.shape != input_ids.shape:
        raise ValueError(f"Token weight shape {tuple(weights.shape)} does not match input_ids {tuple(input_ids.shape)}")
    if config.normalize_mean:
        weights = _normalize_weight_mean(weights, encoded.get("attention_mask"))
    return weights.to(device=device, dtype=torch.float32)


def build_prompt_token_group_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device | str,
) -> list[torch.Tensor]:
    """Build per-token prompt role group ids using the same spans as manual weights."""
    return [build_prompt_token_groups(tokenizer=tokenizer, prompt=prompt, device=device) for prompt in prompts]


def build_prompt_slot_token_group_batches(
    *,
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device | str,
) -> list[torch.Tensor]:
    """Build per-token SID slot group ids: text/sid_a/sid_b/sid_c/boundary."""
    return [build_prompt_slot_token_groups(tokenizer=tokenizer, prompt=prompt, device=device) for prompt in prompts]


def build_prompt_slot_token_weights(
    *,
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
    config: SlotTokenWeightConfig = DEFAULT_SLOT_TOKEN_WEIGHT_CONFIG,
) -> torch.Tensor:
    encoded, offsets = _tokenize_with_offsets(tokenizer, prompt)
    input_ids = _as_tensor(encoded["input_ids"])
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected single-prompt input_ids shape [1, seq], got {tuple(input_ids.shape)}")

    if offsets is None:
        weights = _fallback_slot_weights_from_decoded_tokens(tokenizer, input_ids, config=config)
    else:
        weights = _slot_weights_from_offsets(prompt, offsets, config=config)

    if weights.shape != input_ids.shape:
        raise ValueError(f"Token weight shape {tuple(weights.shape)} does not match input_ids {tuple(input_ids.shape)}")
    if config.normalize_mean:
        weights = _normalize_weight_mean(weights, encoded.get("attention_mask"))
    return weights.to(device=device, dtype=torch.float32)


def build_prompt_token_groups(
    *,
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
) -> torch.Tensor:
    encoded, offsets = _tokenize_with_offsets(tokenizer, prompt)
    input_ids = _as_tensor(encoded["input_ids"])
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected single-prompt input_ids shape [1, seq], got {tuple(input_ids.shape)}")

    if offsets is None:
        groups = _fallback_groups_from_decoded_tokens(tokenizer, input_ids)
    else:
        groups = _groups_from_offsets(prompt, offsets)

    if groups.shape != input_ids.shape:
        raise ValueError(f"Token group shape {tuple(groups.shape)} does not match input_ids {tuple(input_ids.shape)}")
    return groups.to(device=device, dtype=torch.long)


def build_prompt_slot_token_groups(
    *,
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
) -> torch.Tensor:
    encoded, offsets = _tokenize_with_offsets(tokenizer, prompt)
    input_ids = _as_tensor(encoded["input_ids"])
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected single-prompt input_ids shape [1, seq], got {tuple(input_ids.shape)}")

    if offsets is None:
        groups = _fallback_slot_groups_from_decoded_tokens(tokenizer, input_ids)
    else:
        groups = _slot_groups_from_offsets(prompt, offsets)

    if groups.shape != input_ids.shape:
        raise ValueError(f"Token group shape {tuple(groups.shape)} does not match input_ids {tuple(input_ids.shape)}")
    return groups.to(device=device, dtype=torch.long)


def _tokenize_with_offsets(tokenizer: Any, prompt: str) -> tuple[dict[str, Any], torch.Tensor | None]:
    try:
        encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            return dict(encoded), _as_tensor(offsets).long()
    except (NotImplementedError, TypeError, ValueError):
        pass
    encoded = tokenizer(prompt, return_tensors="pt")
    return dict(encoded), None


def _weights_from_offsets(
    prompt: str,
    offsets: torch.Tensor,
    *,
    config: PromptTokenWeightConfig,
) -> torch.Tensor:
    if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[-1] != 2:
        raise ValueError(f"Expected offset_mapping shape [1, seq, 2], got {tuple(offsets.shape)}")
    role_spans = _prompt_role_spans(prompt)
    weights = torch.full((1, offsets.shape[1]), float(config.text_weight), dtype=torch.float32)
    for idx, (start_t, end_t) in enumerate(offsets[0]):
        start = int(start_t.item())
        end = int(end_t.item())
        role = _role_for_span(start, end, role_spans)
        weights[0, idx] = _weight_for_role(role, config)
    return weights


def _groups_from_offsets(prompt: str, offsets: torch.Tensor) -> torch.Tensor:
    if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[-1] != 2:
        raise ValueError(f"Expected offset_mapping shape [1, seq, 2], got {tuple(offsets.shape)}")
    role_spans = _prompt_role_spans(prompt)
    groups = torch.full((1, offsets.shape[1]), PROMPT_TOKEN_GROUP_IDS["text"], dtype=torch.long)
    for idx, (start_t, end_t) in enumerate(offsets[0]):
        start = int(start_t.item())
        end = int(end_t.item())
        role = _role_for_span(start, end, role_spans)
        groups[0, idx] = _group_id_for_role(role)
    return groups


def _slot_weights_from_offsets(
    prompt: str,
    offsets: torch.Tensor,
    *,
    config: SlotTokenWeightConfig,
) -> torch.Tensor:
    if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[-1] != 2:
        raise ValueError(f"Expected offset_mapping shape [1, seq, 2], got {tuple(offsets.shape)}")
    slot_spans = _prompt_slot_spans(prompt)
    weights = torch.full((1, offsets.shape[1]), float(config.text_weight), dtype=torch.float32)
    for idx, (start_t, end_t) in enumerate(offsets[0]):
        start = int(start_t.item())
        end = int(end_t.item())
        role = _role_for_span(start, end, slot_spans)
        weights[0, idx] = _weight_for_slot_role(role, config)
    return weights


def _slot_groups_from_offsets(prompt: str, offsets: torch.Tensor) -> torch.Tensor:
    if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[-1] != 2:
        raise ValueError(f"Expected offset_mapping shape [1, seq, 2], got {tuple(offsets.shape)}")
    slot_spans = _prompt_slot_spans(prompt)
    groups = torch.full((1, offsets.shape[1]), SLOT_TOKEN_GROUP_IDS["text"], dtype=torch.long)
    for idx, (start_t, end_t) in enumerate(offsets[0]):
        start = int(start_t.item())
        end = int(end_t.item())
        role = _role_for_span(start, end, slot_spans)
        groups[0, idx] = _group_id_for_slot_role(role)
    return groups


def _prompt_slot_spans(prompt: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for token in SID_TOKEN_RE.finditer(prompt):
        spans.append((token.start(), token.end(), _slot_role_for_token(token.group(0))))
    return spans


def _slot_role_for_token(token: str) -> str:
    if SID_BOUNDARY_RE.fullmatch(token):
        return "boundary"
    if token.startswith("<s_a_"):
        return "sid_a"
    if token.startswith("<s_b_"):
        return "sid_b"
    if token.startswith("<s_c_"):
        return "sid_c"
    return "text"


def _prompt_role_spans(prompt: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    items = list(SID_ITEM_RE.finditer(prompt))
    groups = _group_sid_items(prompt, items)
    for group_idx, group in enumerate(groups):
        code_role = "history_sid" if group_idx == 0 else "interest_sid"
        for item in group:
            for token in SID_TOKEN_RE.finditer(prompt, item.start(), item.end()):
                role = "sid_boundary" if SID_BOUNDARY_RE.fullmatch(token.group(0)) else code_role
                spans.append((token.start(), token.end(), role))

    for token in SID_BOUNDARY_RE.finditer(prompt):
        if not _overlaps_any(token.start(), token.end(), spans):
            spans.append((token.start(), token.end(), "sid_boundary"))
    return sorted(spans, key=lambda item: (item[0], item[1]))


def _group_sid_items(prompt: str, items: Sequence[re.Match[str]]) -> list[list[re.Match[str]]]:
    if not items:
        return []
    groups: list[list[re.Match[str]]] = [[items[0]]]
    previous = items[0]
    for item in items[1:]:
        gap = prompt[previous.end() : item.start()]
        if SECTION_BREAK_RE.search(SID_TOKEN_RE.sub("", gap)):
            groups.append([item])
        else:
            groups[-1].append(item)
        previous = item
    return groups


def _role_for_span(start: int, end: int, role_spans: Sequence[tuple[int, int, str]]) -> str:
    if end <= start:
        return "text"
    best_role = "text"
    best_overlap = 0
    for span_start, span_end, role in role_spans:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = role
    return best_role


def _weight_for_role(role: str, config: PromptTokenWeightConfig) -> float:
    if role == "history_sid":
        return float(config.history_sid_weight)
    if role == "interest_sid":
        return float(config.interest_sid_weight)
    if role == "sid_boundary":
        return float(config.sid_boundary_weight)
    return float(config.text_weight)


def _group_id_for_role(role: str) -> int:
    if role == "history_sid":
        return PROMPT_TOKEN_GROUP_IDS["history_sid"]
    if role == "interest_sid":
        return PROMPT_TOKEN_GROUP_IDS["interest_sid"]
    if role == "sid_boundary":
        return PROMPT_TOKEN_GROUP_IDS["sid_boundary"]
    return PROMPT_TOKEN_GROUP_IDS["text"]


def _weight_for_slot_role(role: str, config: SlotTokenWeightConfig) -> float:
    if role == "sid_a":
        return float(config.sid_a_weight)
    if role == "sid_b":
        return float(config.sid_b_weight)
    if role == "sid_c":
        return float(config.sid_c_weight)
    if role == "boundary":
        return float(config.boundary_weight)
    return float(config.text_weight)


def _group_id_for_slot_role(role: str) -> int:
    if role == "sid_a":
        return SLOT_TOKEN_GROUP_IDS["sid_a"]
    if role == "sid_b":
        return SLOT_TOKEN_GROUP_IDS["sid_b"]
    if role == "sid_c":
        return SLOT_TOKEN_GROUP_IDS["sid_c"]
    if role == "boundary":
        return SLOT_TOKEN_GROUP_IDS["boundary"]
    return SLOT_TOKEN_GROUP_IDS["text"]


def _fallback_weights_from_decoded_tokens(
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    config: PromptTokenWeightConfig,
) -> torch.Tensor:
    weights = torch.full_like(input_ids, float(config.text_weight), dtype=torch.float32)
    if not hasattr(tokenizer, "decode"):
        return weights
    for idx, token_id in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if SID_BOUNDARY_RE.search(token_text):
            weights[0, idx] = float(config.sid_boundary_weight)
        elif SID_CODE_RE.search(token_text):
            weights[0, idx] = float(config.history_sid_weight)
    return weights


def _fallback_groups_from_decoded_tokens(tokenizer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    groups = torch.full_like(input_ids, PROMPT_TOKEN_GROUP_IDS["text"], dtype=torch.long)
    if not hasattr(tokenizer, "decode"):
        return groups
    for idx, token_id in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if SID_BOUNDARY_RE.search(token_text):
            groups[0, idx] = PROMPT_TOKEN_GROUP_IDS["sid_boundary"]
        elif SID_CODE_RE.search(token_text):
            groups[0, idx] = PROMPT_TOKEN_GROUP_IDS["history_sid"]
    return groups


def _fallback_slot_weights_from_decoded_tokens(
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    config: SlotTokenWeightConfig,
) -> torch.Tensor:
    weights = torch.full_like(input_ids, float(config.text_weight), dtype=torch.float32)
    if not hasattr(tokenizer, "decode"):
        return weights
    for idx, token_id in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if SID_BOUNDARY_RE.search(token_text):
            weights[0, idx] = float(config.boundary_weight)
        elif "<s_a_" in token_text:
            weights[0, idx] = float(config.sid_a_weight)
        elif "<s_b_" in token_text:
            weights[0, idx] = float(config.sid_b_weight)
        elif "<s_c_" in token_text:
            weights[0, idx] = float(config.sid_c_weight)
    return weights


def _fallback_slot_groups_from_decoded_tokens(tokenizer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    groups = torch.full_like(input_ids, SLOT_TOKEN_GROUP_IDS["text"], dtype=torch.long)
    if not hasattr(tokenizer, "decode"):
        return groups
    for idx, token_id in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if SID_BOUNDARY_RE.search(token_text):
            groups[0, idx] = SLOT_TOKEN_GROUP_IDS["boundary"]
        elif "<s_a_" in token_text:
            groups[0, idx] = SLOT_TOKEN_GROUP_IDS["sid_a"]
        elif "<s_b_" in token_text:
            groups[0, idx] = SLOT_TOKEN_GROUP_IDS["sid_b"]
        elif "<s_c_" in token_text:
            groups[0, idx] = SLOT_TOKEN_GROUP_IDS["sid_c"]
    return groups


def _normalize_weight_mean(weights: torch.Tensor, attention_mask: Any) -> torch.Tensor:
    if attention_mask is None:
        mask = torch.ones_like(weights, dtype=torch.bool)
    else:
        mask = _as_tensor(attention_mask).bool()
    if mask.shape != weights.shape:
        raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match weights {tuple(weights.shape)}")
    valid = weights[mask]
    if valid.numel() == 0:
        return weights
    mean = valid.float().mean().clamp_min(1e-12)
    normalized = weights / mean
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def _overlaps_any(start: int, end: int, spans: Sequence[tuple[int, int, str]]) -> bool:
    return any(min(end, span_end) > max(start, span_start) for span_start, span_end, _role in spans)


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)
