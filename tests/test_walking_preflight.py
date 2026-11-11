"""The read-only G1 preflight report."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from inspect_robots_unitree_g1.walking.arm_kinematics import G1_ARMS
from inspect_robots_unitree_g1.walking.config import G1MobileConfig
from inspect_robots_unitree_g1.walking.preflight import ProbeResult, assess, format_report, main

NOMINAL = np.concatenate([G1_ARMS[0].nominal, G1_ARMS[1].nominal])


def healthy(**overrides: Any) -> ProbeResult:
    probe = ProbeResult(
        fsm=(0, 500),
        rpy=np.asarray((0.01, -0.02, 0.3)),
        arm_q=NOMINAL.copy(),
        odom_age_s=0.01,
        odom_pose=np.asarray((0.0, 0.0, 0.3)),
        strokes=np.asarray((5.4, 0.0)),
        frame_shape=(480, 640, 3),
    )
    for key, value in overrides.items():
        setattr(probe, key, value)
    return probe


def levels(cfg: G1MobileConfig, probe: ProbeResult) -> dict[str, str]:
    return {check.name: check.level for check in assess(cfg, probe)}


def test_healthy_robot_is_ready() -> None:
    cfg = G1MobileConfig(hands="dex1")
    checks = assess(cfg, healthy())
    assert all(check.level == "ok" for check in checks)
    report = format_report(checks)
    assert report.endswith("READY for g1_mobile (nothing was sent)")
    assert "grip (0 open, 1 closed) (+0.000, +1.000)" in report


@pytest.mark.parametrize(
    ("overrides", "name", "level", "text"),
    [
        ({"fsm": None}, "locomotion", "fail", "no answer"),
        ({"fsm_error": "Timeout()"}, "locomotion", "fail", "Timeout"),
        ({"fsm": (3104, 0)}, "locomotion", "fail", "code 3104"),
        ({"fsm": (0, 1)}, "locomotion", "fail", "damping"),
        ({"rpy": None}, "lowstate", "fail", "no rt/lowstate"),
        ({"rpy": np.asarray((0.5, 0.0, 0.0))}, "lowstate", "warn", "tilted"),
        ({"arm_q": None}, "arms", "fail", "no arm joint"),
        ({"odom_pose": None}, "odometry", "warn", "dead-reckoned"),
        ({"odom_age_s": 5.0}, "odometry", "warn", "s old"),
        ({"strokes": None}, "grippers", "fail", "powered"),
        ({"strokes": np.asarray((6.0, 0.0))}, "grippers", "warn", "outside 0 to 5.4"),
        ({"camera_error": "Again()"}, "camera", "fail", "Again"),
        ({"frame_shape": None}, "camera", "fail", "no frame"),
    ],
)
def test_each_problem_is_reported(
    overrides: dict[str, Any], name: str, level: str, text: str
) -> None:
    checks = {
        check.name: check for check in assess(G1MobileConfig(hands="dex1"), healthy(**overrides))
    }
    assert checks[name].level == level
    assert text in checks[name].detail


def test_arms_outside_limits_or_far_from_home_are_flagged() -> None:
    cfg = G1MobileConfig()
    q = NOMINAL.copy()
    q[3] = 3.0  # left elbow past its 2.09 rad limit
    check = next(c for c in assess(cfg, healthy(arm_q=q)) if c.name == "arms")
    assert check.level == "warn" and "left_elbow_joint" in check.detail
    hanging = np.zeros(14)
    check = next(c for c in assess(cfg, healthy(arm_q=hanging)) if c.name == "arms")
    assert check.level == "ok"  # arms down, forearms forward: close to home
    lowered = NOMINAL.copy()
    lowered[0] = lowered[7] = 1.5  # shoulders pitched back: hands far from home
    check = next(c for c in assess(cfg, healthy(arm_q=lowered)) if c.name == "arms")
    assert check.level == "warn" and "glide the hands" in check.detail


def test_grippers_are_skipped_without_dex1() -> None:
    assert levels(G1MobileConfig(), healthy(strokes=None))["grippers"] == "ok"


def test_main_prints_and_sets_the_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[tuple[G1MobileConfig, float]] = []

    def probe(cfg: G1MobileConfig, seconds: float) -> ProbeResult:
        seen.append((cfg, seconds))
        return healthy()

    assert main(["-E", "hands=dex1", "-E", "network_interface=enp3s0"], probe_fn=probe) == 0
    cfg, seconds = seen[0]
    assert cfg.backend == "sdk" and cfg.hands == "dex1" and cfg.network_interface == "enp3s0"
    assert seconds == 2.0
    assert "READY" in capsys.readouterr().out

    assert main(["--json", "--seconds", "1"], probe_fn=lambda c, s: healthy(fsm=None)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload[0] == {"name": "locomotion", "level": "fail", "detail": payload[0]["detail"]}


@pytest.mark.parametrize("argv", [["-E", "hands"], ["--seconds", "0"]])
def test_main_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(argv, probe_fn=lambda c, s: healthy())
