"""Read-only G1 preflight: listen to the robot and report, never command it.

Run it on the machine that will drive the G1, with the robot standing in
walking mode, before the first ``g1_mobile`` trial::

    inspect-robots-unitree-g1-robot-check -E network_interface=eth0 -E hands=dex1

It subscribes to the state topics ``g1_mobile`` reads (``rt/lowstate``, the
odometry topic, the Dex1 state topics), asks the locomotion service for its
FSM id (a read-only RPC), and grabs one head camera frame. It creates no
publisher and sends no motion, arm, gripper or mode command. The exit code is
1 when anything that ``g1_mobile`` needs is missing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from inspect_robots_unitree_g1.walking.arm_kinematics import G1_ARMS, fk, with_tool_offset
from inspect_robots_unitree_g1.walking.config import G1MobileConfig
from inspect_robots_unitree_g1.walking.g1 import (
    LIMP_FSM_IDS,
    dex1_stroke_to_grip,
)
from inspect_robots_unitree_g1.walking.mobile import HOME_HANDS

Vec = npt.NDArray[np.float64]

#: Roll or pitch beyond this (radians) means the robot is probably not standing upright.
TILT_WARN_RAD = 0.35
#: Homing at reset glides the palms this far or more: worth a look before the first trial.
HOMING_WARN_M = 0.35
#: Slack on joint limits and Dex1 stroke before a reading is flagged.
LIMIT_SLACK = 0.05


@dataclass
class ProbeResult:
    """What the robot reported during the listening window; ``None`` means nothing arrived."""

    fsm: tuple[int, int] | None = None
    fsm_error: str | None = None
    rpy: Vec | None = None
    arm_q: Vec | None = None
    odom_age_s: float | None = None
    odom_pose: Vec | None = None
    strokes: Vec | None = None
    frame_shape: tuple[int, ...] | None = None
    camera_error: str | None = None


@dataclass(frozen=True)
class Check:
    """One line of the report. ``level`` is ``ok``, ``warn`` or ``fail``."""

    name: str
    level: str
    detail: str


def _valid(values: npt.ArrayLike, size: int) -> bool:
    """Exactly ``size`` finite numbers."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array.shape == (size,) and bool(np.all(np.isfinite(array)))


def _fmt(values: npt.ArrayLike) -> str:
    return "(" + ", ".join(f"{float(v):+.3f}" for v in np.asarray(values).reshape(-1)) + ")"


def _locomotion(probe: ProbeResult) -> Check:
    if probe.fsm_error is not None:
        return Check("locomotion", "fail", f"GetFsmId raised: {probe.fsm_error}")
    if probe.fsm is None:
        return Check("locomotion", "fail", "no answer from the locomotion service")
    code, fsm_id = probe.fsm
    if code != 0:
        return Check("locomotion", "fail", f"GetFsmId returned code {code}")
    if fsm_id in LIMP_FSM_IDS:
        return Check(
            "locomotion",
            "fail",
            f"FSM id {fsm_id} is zero torque or damping; stand it up with the remote",
        )
    return Check("locomotion", "ok", f"FSM id {fsm_id} (not limp)")


def _lowstate(probe: ProbeResult) -> Check:
    if probe.rpy is None:
        return Check("lowstate", "fail", "no rt/lowstate sample")
    if not _valid(probe.rpy, 3):
        return Check("lowstate", "fail", f"invalid IMU reading {probe.rpy!r}")
    roll, pitch = float(probe.rpy[0]), float(probe.rpy[1])
    if max(abs(roll), abs(pitch)) > TILT_WARN_RAD:
        return Check(
            "lowstate", "warn", f"tilted: roll {roll:+.2f}, pitch {pitch:+.2f} rad; upright?"
        )
    return Check("lowstate", "ok", f"IMU rpy {_fmt(probe.rpy)} rad")


def _arms(cfg: G1MobileConfig, probe: ProbeResult) -> Check:
    if probe.arm_q is None:
        return Check("arms", "fail", "no arm joint angles in rt/lowstate")
    if not _valid(probe.arm_q, 14):
        return Check("arms", "fail", "invalid arm joint angles (wrong count or not finite)")
    q = np.asarray(probe.arm_q, dtype=np.float64)
    arms = with_tool_offset(G1_ARMS, cfg.tool_offset_m)
    issues: list[str] = []
    palms = []
    for side, arm in enumerate(arms):
        joints = q[7 * side : 7 * side + 7]
        outside = (joints < arm.lower - LIMIT_SLACK) | (joints > arm.upper + LIMIT_SLACK)
        if np.any(outside):
            names = [arm.joints[i].name for i in np.flatnonzero(outside)]
            issues.append("outside the URDF limits: " + ", ".join(names))
        palms.append(fk(arm, joints))
    travel = float(np.max(np.linalg.norm(HOME_HANDS - np.stack(palms), axis=1)))
    detail = f"palms L {_fmt(palms[0])} R {_fmt(palms[1])} m; homing travel {travel:.2f} m"
    if travel >= HOMING_WARN_M:
        issues.append(f"reset will glide the hands {travel:.2f} m")
    if issues:
        return Check("arms", "warn", detail + "; " + "; ".join(issues))
    return Check("arms", "ok", detail)


def _odometry(cfg: G1MobileConfig, probe: ProbeResult) -> Check:
    if probe.odom_age_s is None or probe.odom_pose is None:
        return Check(
            "odometry", "warn", f"nothing on {cfg.odom_topic}; base_pose will be dead-reckoned"
        )
    if probe.odom_age_s > cfg.odom_stale_s:
        return Check(
            "odometry",
            "warn",
            f"last {cfg.odom_topic} sample is {probe.odom_age_s:.2f} s old; "
            "base_pose will be dead-reckoned",
        )
    if not _valid(probe.odom_pose, 3):
        return Check(
            "odometry", "warn", "odometry pose is not finite; base_pose will be dead-reckoned"
        )
    return Check("odometry", "ok", f"pose {_fmt(probe.odom_pose)}")


def _grippers(cfg: G1MobileConfig, probe: ProbeResult) -> Check:
    if cfg.hands != "dex1":
        return Check("grippers", "ok", f"not used (hands={cfg.hands})")
    if probe.strokes is None:
        return Check("grippers", "fail", "no rt/dex1/*/state samples; are the Dex1s powered?")
    if not _valid(probe.strokes, 2):
        return Check("grippers", "fail", "invalid Dex1 strokes (wrong count or not finite)")
    strokes = np.asarray(probe.strokes, dtype=np.float64)
    grips = dex1_stroke_to_grip(strokes, cfg.dex1_stroke)
    detail = f"strokes {_fmt(strokes)} rad, grip (0 open, 1 closed) {_fmt(grips)}"
    if np.any(strokes < -LIMIT_SLACK) or np.any(strokes > cfg.dex1_stroke + LIMIT_SLACK):
        return Check("grippers", "warn", detail + f"; outside 0 to {cfg.dex1_stroke} rad")
    return Check("grippers", "ok", detail)


def _camera(cfg: G1MobileConfig, probe: ProbeResult) -> Check:
    if probe.camera_error is not None:
        return Check("camera", "fail", f"{cfg.cam_server_address}: {probe.camera_error}")
    if probe.frame_shape is None:
        return Check("camera", "fail", f"no frame from {cfg.cam_server_address}")
    return Check("camera", "ok", f"frame {probe.frame_shape}")


def assess(cfg: G1MobileConfig, probe: ProbeResult) -> list[Check]:
    """Turn what the robot reported into report lines."""
    return [
        _locomotion(probe),
        _lowstate(probe),
        _arms(cfg, probe),
        _odometry(cfg, probe),
        _grippers(cfg, probe),
        _camera(cfg, probe),
    ]


def format_report(checks: list[Check]) -> str:
    """Human-readable report, one line per check, with a verdict last."""
    lines = [f"{check.level.upper():<5} {check.name:<11} {check.detail}" for check in checks]
    failed = [check.name for check in checks if check.level == "fail"]
    lines.append(
        "NOT READY: " + ", ".join(failed) if failed else "READY for g1_mobile (nothing was sent)"
    )
    return "\n".join(lines)


ProbeFn = Callable[[G1MobileConfig, float], ProbeResult]


def _parse_options(items: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"-E expects key=value; got {item!r}")
        options[key] = value
    return options


def main(argv: list[str] | None = None, *, probe_fn: ProbeFn | None = None) -> int:
    """Listen to the robot, print the report, and return 1 if it is not ready."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-unitree-g1-robot-check",
        description="Read-only check of a real G1 before a g1_mobile trial. Sends nothing.",
    )
    parser.add_argument(
        "-E", dest="options", action="append", default=[], help="g1_mobile option key=value"
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="listening window")
    parser.add_argument("--json", action="store_true", help="print the checks as JSON")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    options = _parse_options(args.options)
    options["backend"] = "sdk"
    cfg = G1MobileConfig.from_kwargs(**options)
    probe = (probe_fn or probe_robot)(cfg, args.seconds)
    checks = assess(cfg, probe)
    if args.json:
        payload = [
            {"name": check.name, "level": check.level, "detail": check.detail} for check in checks
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(checks))
    return 1 if any(check.level == "fail" for check in checks) else 0


def probe_robot(cfg: G1MobileConfig, seconds: float) -> ProbeResult:  # pragma: no cover
    """Subscribe, listen for ``seconds``, ask the FSM id, read one frame. No publishers."""
    import threading
    import time

    from inspect_robots_unitree_g1._unitree import _load_unitree

    _load_unitree()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_, SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from inspect_robots_unitree_g1.walking.g1 import (
        ARM_MOTOR_INDICES,
        DEX1_TOPICS,
        _ZmqJpegReader,
    )

    lock = threading.Lock()
    result = ProbeResult()
    odom_time: list[float] = []
    dex: list[float | None] = [None, None]

    def on_lowstate(msg: object) -> None:
        rpy = np.asarray(msg.imu_state.rpy, dtype=np.float64)  # type: ignore[attr-defined]
        q = np.asarray(
            [msg.motor_state[i].q for i in ARM_MOTOR_INDICES],  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        with lock:
            result.rpy, result.arm_q = rpy, q

    def on_odom(msg: object) -> None:
        pose = np.asarray(
            (msg.position[0], msg.position[1], msg.imu_state.rpy[2]),  # type: ignore[attr-defined]
            dtype=np.float64,
        )
        with lock:
            result.odom_pose = pose
            odom_time[:] = [time.monotonic()]

    def on_dex(index: int, msg: object) -> None:
        with lock:
            dex[index] = float(msg.states[0].q)  # type: ignore[attr-defined]

    ChannelFactoryInitialize(0, cfg.network_interface)
    subscribers = [
        ChannelSubscriber("rt/lowstate", LowState_),
        ChannelSubscriber(cfg.odom_topic, SportModeState_),
    ]
    subscribers[0].Init(on_lowstate, 10)
    subscribers[1].Init(on_odom, 10)
    if cfg.hands == "dex1":
        for index, topic in enumerate(DEX1_TOPICS):
            subscriber = ChannelSubscriber(f"{topic}/state", MotorStates_)
            subscriber.Init(lambda msg, i=index: on_dex(i, msg), 10)
            subscribers.append(subscriber)
    try:
        loco = LocoClient()
        loco.SetTimeout(cfg.rpc_timeout_s)
        loco.Init()
        try:
            code, fsm_id = loco.GetFsmId()
            result.fsm = (int(code), int(fsm_id) if fsm_id is not None else -1)
        except Exception as exc:  # report, do not crash the preflight
            result.fsm_error = repr(exc)
        try:
            camera = _ZmqJpegReader(cfg)
            try:
                result.frame_shape = tuple(camera().shape)
            finally:
                camera.close()
        except Exception as exc:
            result.camera_error = repr(exc)
        time.sleep(seconds)
        with lock:
            if odom_time:
                result.odom_age_s = time.monotonic() - odom_time[0]
            if all(value is not None for value in dex):
                result.strokes = np.asarray(dex, dtype=np.float64)
        return result
    finally:
        for subscriber in subscribers:
            subscriber.Close()
