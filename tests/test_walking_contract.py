"""The Go1 contract as the framework, guardrails and LLM agent see it."""

import re

import numpy as np
from inspect_robots.conformance import (
    assert_embodiment_conformant,
    check_evaluation_interface,
    missing_runtime_requirements,
)
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.registry import registered, resolve
from inspect_robots.tools import control_tools

import inspect_robots_unitree_g1.walking as package
from inspect_robots_unitree_g1.walking.config import DIM_LABELS, Go1Config
from inspect_robots_unitree_g1.walking.embodiment import _DOCS, Go1Embodiment


def test_conformance_and_evaluation_interface() -> None:
    emb = Go1Embodiment()
    assert_embodiment_conformant(emb.info)
    assert check_evaluation_interface(emb).ok
    assert missing_runtime_requirements(Go1Embodiment) == {}


def test_construction_is_inert_and_flat_kwargs_work() -> None:
    emb = Go1Embodiment(backend="kinematic", control_hz="5")
    assert emb.info.is_simulated is True
    assert emb.info.control_hz == 5.0
    assert emb.info.environment_id == "go1:kinematic"
    assert Go1Embodiment().info.is_simulated is False
    emb.close()  # nothing connected: a no-op


def test_llm_agent_accepts_the_contract() -> None:
    info = Go1Embodiment().info
    tools = control_tools(info.action_space, info.observation_space, info.control_hz)
    assert tools.library.names == ("move_by", "sleep")
    assert "base_forward: [-0.03, 0.05]" in tools.bounds_text
    schema = tools.library.schemas()[0]["function"]
    assert "base_forward, base_left, base_turn" in str(schema["parameters"])


def test_guardrails_build_from_the_box() -> None:
    DeltaLimitGuard(Go1Embodiment().info.action_space)


def test_docs_name_every_dimension_once() -> None:
    docs = Go1Embodiment().info.docs or ""
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1
    assert Go1Embodiment(Go1Config(docs_extra=" rig note \n")).info.docs == _DOCS + "\n\nrig note"
    assert Go1Embodiment(Go1Config(docs_extra=" \n")).info.docs == _DOCS


def test_public_api_and_entry_point() -> None:
    assert set(package.__all__) == {
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
    }
    assert all(hasattr(package, name) for name in package.__all__)
    assert re.match(r"\d+\.\d+", package.__version__)
    # This package registers the G1 embodiments only; the Go1 class is their base.
    assert resolve("embodiment", "g1_base", backend="kinematic").info.name == "g1_base"
    assert resolve("embodiment", "g1_mobile", backend="kinematic").info.name == "g1_mobile"
    assert "go1_base" not in registered("embodiment")
    emb = Go1Embodiment(backend="kinematic")
    assert emb.info.name == "go1_base"
    assert np.asarray(emb.info.action_space.high).shape == (3,)


def test_runtime_requirements_and_distinct_policy_interfaces() -> None:
    from inspect_robots.conformance import check_policy_interfaces

    from inspect_robots_unitree_g1.embodiment import G1Embodiment as G1Arms
    from inspect_robots_unitree_g1.walking import G1Embodiment, G1MobileEmbodiment

    assert set(G1Embodiment.RUNTIME_REQUIREMENTS) == set(G1Arms.RUNTIME_REQUIREMENTS)
    assert G1MobileEmbodiment.RUNTIME_REQUIREMENTS == G1Embodiment.RUNTIME_REQUIREMENTS
    assert Go1Embodiment.RUNTIME_REQUIREMENTS == {}
    infos = [G1Arms().info, G1Embodiment().info, G1MobileEmbodiment().info]
    assert check_policy_interfaces(infos).ok
