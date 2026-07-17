"""Inspect Robots adapters for Unitree G1 arms and Isaac-GR00T servers.

The package registers embodiment ``g1_arms`` and policy ``gr00t``. Both expose
one shared absolute 16-D arm-and-hand contract and stay inert at construction.
"""

from __future__ import annotations

from inspect_robots_unitree_g1.config import G1Config, Gr00tConfig
from inspect_robots_unitree_g1.embodiment import G1Embodiment
from inspect_robots_unitree_g1.operator import OperatorIO
from inspect_robots_unitree_g1.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM
from inspect_robots_unitree_g1.policy import Gr00tPolicy
from inspect_robots_unitree_g1.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-unitree-g1")
except PackageNotFoundError:  # pragma: no cover - non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [  # noqa: RUF022 - public order is pinned by the accepted plan
    "G1Config",
    "Gr00tConfig",
    "G1Embodiment",
    "Gr00tPolicy",
    "OperatorIO",
    "STATE_KEY",
    "TOTAL_DIM",
    "DIM_LABELS",
    "build",
    "run_preflight",
    "__version__",
]
