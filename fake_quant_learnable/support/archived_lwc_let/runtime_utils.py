from __future__ import annotations

from fake_quant_learnable.support.runtime_utils import (
    _detach_tree,
    _module_device,
    _move_tree_to_device,
)

__all__ = ["_detach_tree", "_module_device", "_move_tree_to_device"]
