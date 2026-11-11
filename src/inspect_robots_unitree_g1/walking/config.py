"""Frozen walking-robot configurations (Go1, G1) and the spaces derived from them.

The action is a per-control-step body displacement ``(forward m, left m,
turn rad)``. It is declared as ``joint_delta`` because Inspect Robots has no
base-motion control mode yet; the three "joints" are the body's planar
degrees of freedom, and the operating notes tell policies so. The embodiment
converts each displacement to a body velocity held for one control period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal, Self

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

Backend = Literal["sdk", "kinematic", "isaaclab"]
BACKENDS: tuple[str, ...] = ("sdk", "kinematic", "isaaclab")

#: Action dimension names, in order. Positive forward is the nose direction,
#: positive left is the robot's left, positive turn is counter-clockwise seen
#: from above.
DIM_LABELS: tuple[str, str, str] = ("base_forward", "base_left", "base_turn")
ACTION_DIM = len(DIM_LABELS)

#: Proprioception keys. ``base_pose`` is odometry ``(x m, y m, yaw rad)``
#: relative to where the robot stood at the last reset, ``base_vel`` is the
#: measured body velocity ``(forward m/s, left m/s, turn rad/s)`` and
#: ``body_rpy`` is the IMU roll, pitch and yaw in radians.
POSE_KEY = "base_pose"
VEL_KEY = "base_vel"
RPY_KEY = "body_rpy"

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_delta",
    frame="base",
    dim_labels=DIM_LABELS,
)

STATE_SPEC = StateSpec(
    fields=(
        StateField(key=POSE_KEY, shape=(3,), unit="m,m,rad"),
        StateField(key=VEL_KEY, shape=(3,), unit="m/s,m/s,rad/s"),
        StateField(key=RPY_KEY, shape=(3,), unit="rad"),
    )
)


@dataclass(frozen=True)
class LeggedConfig:
    """Settings shared by every walking robot. Every field is a scalar so ``-E key=value`` works.

    Speed caps bound the action box: the per-step displacement limit is the
    speed cap divided by ``control_hz``. ``command_timeout_s`` is the watchdog
    grace after a step's control period ends; when no new step arrives in that
    window (for example while an LLM is thinking) the commanded velocity drops
    to zero. ``geofence_m`` ends the trial and stops the robot once odometry
    puts it farther than this from the reset point; ``None`` disables it.
    Subclasses set robot-specific defaults and add their transport fields.
    """

    #: Isaac Lab camera mount: the body link and its offset in metres.
    ISAAC_CAMERA_LINK: ClassVar[str] = "trunk"
    ISAAC_CAMERA_POS: ClassVar[tuple[float, float, float]] = (0.28, 0.0, 0.05)

    backend: Backend = "sdk"
    control_hz: float = 10.0
    stream_hz: float = 50.0
    max_forward_speed: float = 0.5
    max_backward_speed: float = 0.3
    max_lateral_speed: float = 0.3
    max_turn_rate: float = 1.0
    command_timeout_s: float = 0.1
    geofence_m: float | None = 3.0
    camera_name: str = "front_cam"
    camera_height: int = 480
    camera_width: int = 640
    unattended: bool = False
    docs_extra: str = ""
    # Isaac Lab backend.
    isaac_task: str = "Isaac-Velocity-Flat-Unitree-Go1-Play-v0"
    isaac_checkpoint: str = ""
    headless: bool = True
    #: Place coloured boxes in the Isaac scene so camera-driven policies have targets.
    isaac_props: bool = True

    def make_action_box(self) -> Box:
        """The action space this config describes."""
        return action_box(self)

    def make_observation_space(self) -> ObservationSpace:
        """The observation space this config describes."""
        return observation_space(self)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}; got {self.backend!r}")
        for name in (
            "control_hz",
            "stream_hz",
            "max_forward_speed",
            "max_backward_speed",
            "max_lateral_speed",
            "max_turn_rate",
        ):
            _require_positive(name, getattr(self, name))
        if self.stream_hz < self.control_hz:
            raise ValueError("stream_hz must be >= control_hz")
        if not math.isfinite(self.command_timeout_s) or self.command_timeout_s < 0:
            raise ValueError("command_timeout_s must be finite and >= 0")
        if self.geofence_m is not None:
            _require_positive("geofence_m", self.geofence_m)
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError("camera_height and camera_width must be > 0")
        if not self.camera_name.strip():
            raise ValueError("camera_name must be non-empty")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Build from CLI-style kwargs, coercing strings and rejecting unknown keys."""
        known = {item.name: item for item in fields(cls)}
        unknown = sorted(set(kwargs) - set(known))
        if unknown:
            raise TypeError(f"unknown {cls.__name__} option(s): {', '.join(unknown)}")
        coerced: dict[str, Any] = {}
        for key, value in kwargs.items():
            coerced[key] = _coerce(key, value, cls.__dataclass_fields__[key].default)
        return cls(**coerced)

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Per-step displacement lower bounds ``(back m, right m, clockwise rad)``."""
        return -np.asarray(
            (self.max_backward_speed, self.max_lateral_speed, self.max_turn_rate),
            dtype=np.float64,
        ) / float(self.control_hz)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Per-step displacement upper bounds ``(forward m, left m, counter-clockwise rad)``."""
        return np.asarray(
            (self.max_forward_speed, self.max_lateral_speed, self.max_turn_rate),
            dtype=np.float64,
        ) / float(self.control_hz)


@dataclass(frozen=True)
class Go1Config(LeggedConfig):
    """Unitree Go1 quadruped over the legacy ``unitree_legged_sdk`` high-level UDP link."""

    robot_ip: str = "192.168.123.161"
    robot_port: int = 8082
    local_port: int = 8080


@dataclass(frozen=True)
class G1Config(LeggedConfig):
    """Unitree G1 humanoid walking under its built-in locomotion controller.

    Speed caps start well below the controller's limits: a humanoid that trips
    falls. ``stream_hz`` is lower than the Go1's because each command is a DDS
    RPC round trip. ``velocity_hold_s`` is the duration sent with every
    velocity command; the robot itself stops when it lapses, so it stops even
    if this process is killed outright.
    """

    ISAAC_CAMERA_LINK: ClassVar[str] = "torso_link"
    ISAAC_CAMERA_POS: ClassVar[tuple[float, float, float]] = (0.08, 0.0, 0.45)

    stream_hz: float = 20.0
    max_forward_speed: float = 0.3
    max_backward_speed: float = 0.2
    max_lateral_speed: float = 0.2
    max_turn_rate: float = 0.5
    camera_name: str = "head_cam"
    isaac_task: str = "Isaac-Velocity-Flat-G1-Play-v0"
    # Real robot (unitree_sdk2py, DDS on the robot's 192.168.123.0/24 network).
    network_interface: str = "eth0"
    velocity_hold_s: float = 0.3
    rpc_timeout_s: float = 0.5
    odom_topic: str = "rt/odommodestate"
    odom_stale_s: float = 0.5
    cam_server_address: str = "tcp://192.168.123.164:5556"
    cam_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("velocity_hold_s", "rpc_timeout_s", "odom_stale_s", "cam_timeout_s"):
            _require_positive(name, getattr(self, name))
        if self.velocity_hold_s * self.stream_hz < 2.0:
            raise ValueError(
                "velocity_hold_s must cover at least two stream periods "
                "(velocity_hold_s * stream_hz >= 2)"
            )
        for name in ("network_interface", "odom_topic", "cam_server_address"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")


#: Walking plus both hands: per-step body displacement, per-step palm
#: displacement in the torso frame, and per-step grip change.
MOBILE_DIM_LABELS: tuple[str, ...] = (
    *DIM_LABELS,
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "left_grip",
    "right_grip",
)
HANDS_KEY = "hand_pos"
GRIPS_KEY = "grip"

MOBILE_ACTION_SEMANTICS = ActionSemantics(
    control_mode="eef_delta_pos",
    frame="base",
    gripper="continuous",
    dim_labels=MOBILE_DIM_LABELS,
)


@dataclass(frozen=True)
class G1MobileConfig(G1Config):
    """The G1 walking and using both arms in one trial.

    Hands are palm positions in the torso frame (x forward, y left, z up, metres
    from ``torso_link``), moved by small per-step displacements and clamped to a
    workspace box. ``hand_speed`` bounds each palm's per-axis speed. Grips are
    0 (open) to 1 (closed); ``grip_rate`` is the fastest change per second.
    ``hands`` is ``"auto"`` (grips work in simulation and are pinned at 0 on the
    real robot), ``"dex1"`` (Unitree Dex1-1 two-finger grippers on the real
    robot, and grips in simulation) or ``"none"`` (grips always pinned).
    ``arm_joint_step`` is the largest change per stream tick of any arm joint
    command on the real robot (radians). ``tool_offset_m`` moves the
    controlled hand point that far past the palm along the forearm, for a
    gripper whose grasp point sits beyond the palm; 0 controls the palm.
    """

    hand_speed: float = 0.15
    grip_rate: float = 2.0
    hands: str = "auto"
    hand_x_min: float = 0.18
    hand_x_max: float = 0.40
    hand_y_inner: float = 0.05
    hand_y_outer: float = 0.38
    hand_z_min: float = -0.05
    hand_z_max: float = 0.30
    # Real robot arm control (rt/arm_sdk), as in g1_arms.
    arm_weight_ramp_s: float = 2.0
    arm_joint_step: float = 0.05
    kp_arm: float = 60.0
    kd_arm: float = 1.5
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5
    tool_offset_m: float = 0.0
    #: isaaclab backend with hands=dex1: a G1 USD with Dex1-1 grippers,
    #: converted from unitree_ros g1_29dof_mode_15_with_dex1_1.urdf.
    isaac_robot_usd: str = ""
    # Dex1-1 gripper, the same figures g1_arms uses (config.py, packing.py):
    # xr_teleoperate@7dc9aa1a6edb robot_hand_unitree.py records 0 rad as
    # closed, 5.4 rad as open, and kp 5.0, kd 0.05. The 2.7 rad/s speed cap
    # is the conservative choice g1_arms makes.
    dex1_stroke: float = 5.4
    dex1_max_speed: float = 2.7
    dex1_kp: float = 5.0
    dex1_kd: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "dex1_stroke",
            "dex1_max_speed",
            "dex1_kp",
            "dex1_kd",
            "hand_speed",
            "grip_rate",
            "arm_weight_ramp_s",
            "arm_joint_step",
            "kp_arm",
            "kd_arm",
            "kp_wrist",
            "kd_wrist",
        ):
            _require_positive(name, getattr(self, name))
        if self.hands not in ("auto", "dex1", "none"):
            raise ValueError(f"hands must be 'auto', 'dex1' or 'none'; got {self.hands!r}")
        if not math.isfinite(self.tool_offset_m) or not 0.0 <= self.tool_offset_m <= 0.25:
            raise ValueError("tool_offset_m must be between 0 and 0.25 m")
        if not (self.hand_x_min < self.hand_x_max and self.hand_z_min < self.hand_z_max):
            raise ValueError("hand workspace min must be below max")
        if not 0.0 <= self.hand_y_inner < self.hand_y_outer:
            raise ValueError("need 0 <= hand_y_inner < hand_y_outer")

    @property
    def grips_enabled(self) -> bool:
        """Grips are commandable in simulation, and on the real robot with Dex1 grippers."""
        if self.hands == "dex1":
            return True
        return self.hands == "auto" and self.backend != "sdk"

    @property
    def hand_box(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Workspace bounds, shape ``(2, 3)`` each: rows are left and right palms."""
        low = np.asarray(
            (
                (self.hand_x_min, self.hand_y_inner, self.hand_z_min),
                (self.hand_x_min, -self.hand_y_outer, self.hand_z_min),
            )
        )
        high = np.asarray(
            (
                (self.hand_x_max, self.hand_y_outer, self.hand_z_max),
                (self.hand_x_max, -self.hand_y_inner, self.hand_z_max),
            )
        )
        return low, high

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Per-step lower bounds: walking, then six hand axes, then two grips."""
        hand = -self.hand_speed / self.control_hz
        grip = -self.grip_rate / self.control_hz if self.grips_enabled else 0.0
        return np.concatenate((super().low, np.full(6, hand), np.full(2, grip)))

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Per-step upper bounds: walking, then six hand axes, then two grips."""
        hand = self.hand_speed / self.control_hz
        grip = self.grip_rate / self.control_hz if self.grips_enabled else 0.0
        return np.concatenate((super().high, np.full(6, hand), np.full(2, grip)))

    def make_action_box(self) -> Box:
        """The 11-D walking-and-hands displacement box."""
        return Box(
            shape=(len(MOBILE_DIM_LABELS),),
            low=self.low,
            high=self.high,
            semantics=MOBILE_ACTION_SEMANTICS,
        )

    def make_observation_space(self) -> ObservationSpace:
        """Base state plus palm positions and grips."""
        base = observation_space(self)
        assert base.state is not None
        return ObservationSpace(
            cameras=base.cameras,
            state=StateSpec(
                fields=(
                    *base.state.fields,
                    StateField(key=HANDS_KEY, shape=(6,), unit="m"),
                    StateField(key=GRIPS_KEY, shape=(2,), unit="normalized"),
                )
            ),
        )


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


_NONE_WORDS = frozenset({"none", "null", ""})
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _coerce(key: str, value: Any, default: Any) -> Any:
    """Coerce a CLI string to the type implied by the field's default."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if key == "geofence_m":
        return None if text.lower() in _NONE_WORDS else float(text)
    if isinstance(default, bool):
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ValueError(f"{key} must be a boolean; got {value!r}")
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return value


def action_box(cfg: LeggedConfig | None = None) -> Box:
    """Build the 3-D per-step body displacement action space."""
    resolved = cfg if cfg is not None else Go1Config()
    return Box(
        shape=(ACTION_DIM,), low=resolved.low, high=resolved.high, semantics=ACTION_SEMANTICS
    )


def observation_space(cfg: LeggedConfig | None = None) -> ObservationSpace:
    """Build the one-camera, odometry-and-IMU observation contract."""
    resolved = cfg if cfg is not None else Go1Config()
    return ObservationSpace(
        cameras=(
            CameraSpec(
                name=resolved.camera_name,
                height=resolved.camera_height,
                width=resolved.camera_width,
                channels=3,
            ),
        ),
        state=STATE_SPEC,
    )
