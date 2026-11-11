"""The walking G1: ``g1_base`` (walking) and ``g1_mobile`` (walking plus both hands).

The G1 embodiments share their stepping, streaming and safety layer with a
Go1 quadruped embodiment (``Go1Embodiment``), kept here as their base class.
Walking is a 3-D per-step body displacement ``(base_forward m, base_left m,
base_turn rad)`` executed through the robot's own walking controller.
Construction is inert; backends connect on the first ``reset()``.
"""

from __future__ import annotations

from inspect_robots_unitree_g1 import __version__
from inspect_robots_unitree_g1.walking.config import (
    ACTION_DIM,
    DIM_LABELS,
    G1Config,
    G1MobileConfig,
    Go1Config,
    LeggedConfig,
)
from inspect_robots_unitree_g1.walking.driver import Go1Driver, Go1State, KinematicGo1Driver
from inspect_robots_unitree_g1.walking.embodiment import G1Embodiment, Go1Embodiment
from inspect_robots_unitree_g1.walking.g1 import G1LocoDriver, G1MobileSdkDriver
from inspect_robots_unitree_g1.walking.mobile import G1MobileEmbodiment, KinematicMobileDriver
from inspect_robots_unitree_g1.walking.operator import OperatorIO

__all__ = [
    "ACTION_DIM",
    "DIM_LABELS",
    "G1Config",
    "G1Embodiment",
    "G1LocoDriver",
    "G1MobileConfig",
    "G1MobileEmbodiment",
    "G1MobileSdkDriver",
    "Go1Config",
    "Go1Driver",
    "Go1Embodiment",
    "Go1State",
    "KinematicGo1Driver",
    "KinematicMobileDriver",
    "LeggedConfig",
    "OperatorIO",
    "__version__",
]
