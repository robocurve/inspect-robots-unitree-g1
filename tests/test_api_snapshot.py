import re

import inspect_robots_unitree_g1 as package

EXPECTED = {
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
}


def test_public_api_is_exact_and_versioned() -> None:
    assert set(package.__all__) == EXPECTED
    assert all(hasattr(package, name) for name in package.__all__)
    assert re.match(r"\d+\.\d+", package.__version__)


def test_entry_points_resolve() -> None:
    from inspect_robots.registry import resolve

    assert resolve("policy", "gr00t").info.name == "gr00t"
    assert resolve("embodiment", "g1_arms").info.name == "g1_arms"
