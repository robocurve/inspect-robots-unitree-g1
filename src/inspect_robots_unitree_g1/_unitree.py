"""Load the optional Unitree SDK lazily with actionable installation guidance."""

from __future__ import annotations

from typing import Any

UNITREE_INSTALL_COMMAND = (
    'pip install "unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python"'
)


def _load_unitree() -> Any:
    """Import the git-only SDK or explain its DDS and network prerequisites."""
    try:
        import unitree_sdk2py
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py" and not (exc.name or "").startswith("unitree_sdk2py."):
            raise
        raise ModuleNotFoundError(
            "unitree_sdk2py is the git-only Unitree hardware SDK. Install it with: "
            f"{UNITREE_INSTALL_COMMAND}. Known-good: Python 3.12, unitree_sdk2py 1.0.1, "
            "cyclonedds 0.10.2. If the CycloneDDS wheel is unavailable, build CycloneDDS "
            "from source, set CYCLONEDDS_HOME, then reinstall cyclonedds. The adapter must "
            "run on PC2 or a Linux host with L2 adjacency to 192.168.123.0/24.",
            name=exc.name,
        ) from exc
    return unitree_sdk2py
