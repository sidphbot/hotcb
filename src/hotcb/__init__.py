"""
hotcb unified runtime.

This package introduces a shared HotKernel that routes control-plane updates
to module controllers (callbacks, optimizers, losses) and records an applied
ledger for deterministic replay.
"""

from .kernel import HotKernel  # noqa: F401
from .ops import HotOp  # noqa: F401
