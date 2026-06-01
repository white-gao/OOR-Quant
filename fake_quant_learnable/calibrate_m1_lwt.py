from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .apply import learnable_lwt_parameters


Batch = torch.Tensor | Sequence[Any] | Mapping[str, Any]

DEFAULT_LWT_LR = 3e-4
DEFAULT_LET_LR = 6e-4
DEFAULT_EPOCHS = 2


@dataclass(frozen=True)
class CalibrationHistory:
    initial_loss: float
    final_loss: float
    losses: list[float]


def calibrate_block_mse(
    *,
    teacher_block: nn.Module,
    quant_block: nn.Module,
    batches: Iterable[Batch],
    target_batches: Iterable[Batch] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    lwt_lr: float = DEFAULT_LWT_LR,
    let_lr: float = DEFAULT_LET_LR,
    eps: float = 1e-12,
    max_grad_norm: float | None = 1.0,
    train_lwt: bool = True,
    train_let: bool = True,
) -> CalibrationHistory:
    """Optimize learnable LWT/LET parameters with block-output reconstruction MSE."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    quant_batch_list = list(batches)
    if not quant_batch_list:
        raise ValueError("batches must contain at least one calibration batch.")
    target_batch_list = quant_batch_list if target_batches is None else list(target_batches)
    if len(target_batch_list) != len(quant_batch_list):
        raise ValueError(
            "target_batches and batches must contain the same number of calibration batches: "
            f"got {len(target_batch_list)} and {len(quant_batch_list)}."
        )

    all_params = list(learnable_lwt_parameters(quant_block))
    lwt_params = list(
        learnable_lwt_parameters(
            quant_block,
            include_lwt=train_lwt,
            include_let=False,
        )
    )
    let_params = list(
        learnable_lwt_parameters(
            quant_block,
            include_lwt=False,
            include_let=train_let,
        )
    )
    params = _dedupe_parameters([*lwt_params, *let_params])
    if not params:
        raise ValueError("quant_block does not contain selected learnable quantization parameters.")

    selected_param_ids = {id(param) for param in params}
    requires_grad_state = [(param, param.requires_grad) for param in all_params]
    for param in all_params:
        param.requires_grad_(id(param) in selected_param_ids)

    teacher_was_training = teacher_block.training
    quant_was_training = quant_block.training
    teacher_block.eval()
    quant_block.eval()

    initial_loss = evaluate_block_mse(
        teacher_block=teacher_block,
        quant_block=quant_block,
        batches=quant_batch_list,
        target_batches=target_batch_list,
        eps=eps,
    )

    param_groups = []
    if lwt_params:
        param_groups.append({"params": lwt_params, "lr": lwt_lr})
    if let_params:
        param_groups.append({"params": let_params, "lr": let_lr})
    optimizer = torch.optim.Adam(param_groups)
    losses: list[float] = []
    for target_batch, quant_batch in _iter_shuffled_epoch_batch_pairs(
        target_batch_list,
        quant_batch_list,
        epochs,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = block_mse_loss(
            teacher_block=teacher_block,
            quant_block=quant_block,
            target_batch=target_batch,
            quant_batch=quant_batch,
            eps=eps,
        )
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    final_loss = evaluate_block_mse(
        teacher_block=teacher_block,
        quant_block=quant_block,
        batches=quant_batch_list,
        target_batches=target_batch_list,
        eps=eps,
    )

    for param, requires_grad in requires_grad_state:
        param.requires_grad_(requires_grad)

    teacher_block.train(teacher_was_training)
    quant_block.train(quant_was_training)
    return CalibrationHistory(
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=losses,
    )


def _dedupe_parameters(params: Sequence[nn.Parameter]) -> list[nn.Parameter]:
    deduped: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        deduped.append(param)
    return deduped


def evaluate_block_mse(
    *,
    teacher_block: nn.Module,
    quant_block: nn.Module,
    batches: Iterable[Batch],
    target_batches: Iterable[Batch] | None = None,
    eps: float = 1e-12,
) -> float:
    quant_batch_list = list(batches)
    if not quant_batch_list:
        raise ValueError("batches must contain at least one calibration batch.")
    target_batch_list = quant_batch_list if target_batches is None else list(target_batches)
    if len(target_batch_list) != len(quant_batch_list):
        raise ValueError(
            "target_batches and batches must contain the same number of calibration batches: "
            f"got {len(target_batch_list)} and {len(quant_batch_list)}."
        )

    losses: list[float] = []
    with torch.no_grad():
        for target_batch, quant_batch in zip(target_batch_list, quant_batch_list):
            loss = block_mse_loss(
                teacher_block=teacher_block,
                quant_block=quant_block,
                target_batch=target_batch,
                quant_batch=quant_batch,
                eps=eps,
            )
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _iter_shuffled_epoch_batch_pairs(
    target_batch_list: Sequence[Batch],
    quant_batch_list: Sequence[Batch],
    epochs: int,
) -> Iterable[tuple[Batch, Batch]]:
    """Yield paired FP-target and quant-input batches once per shuffled epoch."""
    num_batches = len(quant_batch_list)
    if num_batches <= 0:
        raise ValueError("batch_list must contain at least one calibration batch.")
    if len(target_batch_list) != num_batches:
        raise ValueError(
            "target_batch_list and quant_batch_list must contain the same number of batches: "
            f"got {len(target_batch_list)} and {num_batches}."
        )

    for _ in range(epochs):
        for idx in torch.randperm(num_batches).tolist():
            yield target_batch_list[idx], quant_batch_list[idx]


def block_mse_loss(
    *,
    teacher_block: nn.Module,
    quant_block: nn.Module,
    batch: Batch | None = None,
    target_batch: Batch | None = None,
    quant_batch: Batch | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if batch is not None:
        if target_batch is not None or quant_batch is not None:
            raise ValueError("Pass either batch or target_batch/quant_batch, not both.")
        target_batch = batch
        quant_batch = batch
    if target_batch is None or quant_batch is None:
        raise ValueError("target_batch and quant_batch must both be provided.")

    target_args, target_kwargs = _batch_to_args_kwargs(target_batch)
    quant_args, quant_kwargs = _batch_to_args_kwargs(quant_batch)
    with torch.no_grad():
        target = _first_tensor(teacher_block(*target_args, **target_kwargs)).detach()
    pred = _first_tensor(quant_block(*quant_args, **quant_kwargs))
    diff = (pred.float() - target.float()).pow(2).mean()
    denom = target.float().pow(2).mean().clamp_min(eps)
    return diff / denom


def _batch_to_args_kwargs(batch: Batch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if torch.is_tensor(batch):
        return (batch,), {}
    if isinstance(batch, Mapping):
        return (), dict(batch)
    if isinstance(batch, Sequence):
        if len(batch) == 2 and isinstance(batch[1], Mapping) and isinstance(batch[0], Sequence):
            return tuple(batch[0]), dict(batch[1])
        return tuple(batch), {}
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _first_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, Mapping):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Block output does not contain a tensor: {type(output)!r}")
