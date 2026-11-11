"""The walking G1: ``g1_base`` (walking) and ``g1_mobile`` (walking plus both hands).

Both build on ``LeggedEmbodiment``, the shared stepping, streaming and safety
layer for a robot walked through its own locomotion controller.
Walking is a 3-D per-step body displacement ``(base_forward m, base_left m,
base_turn rad)`` executed through the robot's own walking controller.
Construction is inert; backends connect on the first ``reset()``.
"""

from __future__ import annotations

from inspect_robots_unitree_g1 import __version__
from inspect_robots_unitree_g1.operator import OperatorIO
from inspect_robots_unitree_g1.walking.config import (
    ACTION_DIM,
    DIM_LABELS,
    G1Config,
    G1MobileConfig,
    LeggedConfig,
)
from inspect_robots_unitree_g1.walking.driver import (
    KinematicLeggedDriver,
    LeggedDriver,
    LeggedState,
)
from inspect_robots_unitree_g1.walking.embodiment import G1Embodiment, LeggedEmbodiment
from inspect_robots_unitree_g1.walking.g1 import G1LocoDriver, G1MobileSdkDriver
from inspect_robots_unitree_g1.walking.mobile import G1MobileEmbodiment, KinematicMobileDriver

__all__ = [
    "ACTION_DIM",
    "DIM_LABELS",
    "G1Config",
    "G1Embodiment",
    "G1LocoDriver",
    "G1MobileConfig",
    "G1MobileEmbodiment",
    "G1MobileSdkDriver",
    "KinematicLeggedDriver",
    "KinematicMobileDriver",
    "LeggedConfig",
    "LeggedConfig",
    "LeggedDriver",
    "LeggedEmbodiment",
    "LeggedState",
    "OperatorIO",
    "__version__",
]
