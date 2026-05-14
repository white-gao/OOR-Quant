from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import (
    ActQuant,
    fp8_activation_per_token,
    fp8_activation_static_tensor,
    fp8_weight_per_channel,
)


class FakeQuantLinear(nn.Module):
    """Linear layer with cached FP8 weight QDQ and optional activation QDQ."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_quant: ActQuant = "none",
        activation_static_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if act_quant not in ("none", "per_token", "static_tensor"):
            raise ValueError(f"Unsupported act_quant: {act_quant}")
        if act_quant == "static_tensor" and activation_static_scale is None:
            raise ValueError("activation_static_scale is required for act_quant='static_tensor'.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.act_quant = act_quant

        with torch.no_grad():
            weight_qdq = fp8_weight_per_channel(linear.weight.detach())

        self.register_buffer("weight_qdq", weight_qdq, persistent=True)
        if activation_static_scale is not None:
            self.register_buffer(
                "activation_static_scale",
                activation_static_scale.detach().float().reshape(()).to(linear.weight.device),
                persistent=True,
            )
        else:
            self.activation_static_scale = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant == "per_token":
            x = fp8_activation_per_token(x)
        elif self.act_quant == "static_tensor":
            x = fp8_activation_static_tensor(x, self.activation_static_scale)
        return F.linear(x, self.weight_qdq, self.bias)
