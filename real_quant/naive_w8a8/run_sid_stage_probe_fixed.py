"""Entry point for the corrected shared-input stage rescue probe."""

from __future__ import annotations

from . import run_sid_stage_probe as _probe
from .stage_probe_runtime_fixed import activate_stage_activation_rescue


_probe.activate_stage_activation_rescue = activate_stage_activation_rescue


if __name__ == "__main__":
    _probe.main()
