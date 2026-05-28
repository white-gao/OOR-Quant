from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .apply import learnable_lwt_parameters


Batch = torch.Tensor | Sequence[Any] | Mapping[str, Any]


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
    steps: int = 200,
    lr: float = 1e-3,
    eps: float = 1e-12,
    max_grad_norm: float | None = 1.0,
    train_lwt: bool = True,
    train_let: bool = True,
) -> CalibrationHistory:
    """Optimize M1 LWT parameters with plain block-output reconstruction MSE."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("batches must contain at least one calibration batch.")

    all_params = list(learnable_lwt_parameters(quant_block))
    params = list(
        learnable_lwt_parameters(
            quant_block,
            include_lwt=train_lwt,
            include_let=train_let,
        )
    )
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
        batches=batch_list,
        eps=eps,
    )

    optimizer = torch.optim.Adam(params, lr=lr)
    losses: list[float] = []
    for batch, _ in zip(cycle(batch_list), range(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = block_mse_loss(
            teacher_block=teacher_block,
            quant_block=quant_block,
            batch=batch,
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
        batches=batch_list,
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


def evaluate_block_mse(
    *,
    teacher_block: nn.Module,
    quant_block: nn.Module,
    batches: Iterable[Batch],
    eps: float = 1e-12,
) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            loss = block_mse_loss(
                teacher_block=teacher_block,
                quant_block=quant_block,
                batch=batch,
                eps=eps,
            )
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("batches must contain at least one calibration batch.")
    return sum(losses) / len(losses)


def block_mse_loss(
    *,
    teacher_block: nn.Module,
    quant_block: nn.Module,
    batch: Batch,
    eps: float = 1e-12,
) -> torch.Tensor:
    args, kwargs = _batch_to_args_kwargs(batch)
    with torch.no_grad():
        target = _first_tensor(teacher_block(*args, **kwargs)).detach()
    pred = _first_tensor(quant_block(*args, **kwargs))
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
