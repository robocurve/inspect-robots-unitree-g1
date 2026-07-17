"""Validated configuration and shared spaces for the G1 and GR00T pair."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.errors import ConfigError
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

from inspect_robots_unitree_g1.packing import (
    DIM_LABELS,
    STATE_KEY,
    TOTAL_DIM,
)

_T = TypeVar("_T", bound="_FromKwargs")

PackedSpec: TypeAlias = tuple[Literal["packed"], int, int]
HandSpec: TypeAlias = tuple[Literal["hand"], Literal["left", "right"]]
StateSourceSpec: TypeAlias = tuple[Literal["state"], str]
ConstSpec: TypeAlias = tuple[Literal["const"], int, float]
SourceSpec: TypeAlias = PackedSpec | HandSpec | StateSourceSpec | ConstSpec
ActionSpec: TypeAlias = PackedSpec | HandSpec

# Raw limits from unitreerobotics/unitree_ros@d96d8f63ae17,
# robots/g1_description/g1_29dof.urdf. Defaults pull each bound 0.05 rad inward.
_RAW_ARM_LOW = (
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.614429558,
    -1.614429558,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.614429558,
    -1.614429558,
)
_RAW_ARM_HIGH = (
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.972222054,
    1.614429558,
    1.614429558,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.972222054,
    1.614429558,
    1.614429558,
)


def _with_hands(arms: tuple[float, ...], hand: float) -> tuple[float, ...]:
    return (*arms[:7], hand, *arms[7:], hand)


DEFAULT_JOINT_LOW = _with_hands(tuple(value + 0.05 for value in _RAW_ARM_LOW), 0.0)
DEFAULT_JOINT_HIGH = _with_hands(tuple(value - 0.05 for value in _RAW_ARM_HIGH), 1.0)
# unitree_sdk2_python@e4cd91f051aa,
# example/g1/high_level/g1_arm7_sdk_dds_example.py returns all arm joints to zero.
DEFAULT_HOME_POSE = _with_hands((0.0,) * 14, 1.0)

DEFAULT_STATE_KEYS: Mapping[str, SourceSpec] = {
    "state.left_arm": ("packed", 0, 7),
    "state.right_arm": ("packed", 8, 15),
    "state.left_hand": ("hand", "left"),
    "state.right_hand": ("hand", "right"),
    "state.left_leg": ("state", "left_leg"),
    "state.right_leg": ("state", "right_leg"),
    "state.waist": ("state", "waist"),
}
DEFAULT_ACTION_KEYS: Mapping[str, ActionSpec] = {
    "action.left_arm": ("packed", 0, 7),
    "action.right_arm": ("packed", 8, 15),
    "action.left_hand": ("hand", "left"),
    "action.right_hand": ("hand", "right"),
}

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
    dim_labels=DIM_LABELS,
)

# Shapes follow unitree_sdk2_python@e4cd91f051aa G1JointIndex: legs 0-5 and
# 6-11, waist 12-14. The N1.7 key set is Isaac-GR00T@9c7e746b2cd37a,
# gr00t/configs/data/embodiment_configs.py.
STATE_SPEC = StateSpec(
    fields=(
        StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),
        StateField(key="left_leg", shape=(6,), unit="rad"),
        StateField(key="right_leg", shape=(6,), unit="rad"),
        StateField(key="waist", shape=(3,), unit="rad"),
    )
)


class _FromKwargs:
    """Build frozen dataclasses from flat CLI-friendly keyword arguments."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        """Reject unknown keys and parse configured comma-separated tuples."""
        names = {item.name for item in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        for key in cls._FLOAT_TUPLE_FIELDS & set(flat):
            value = flat[key]
            if isinstance(value, str):
                try:
                    flat[key] = tuple(float(part) for part in value.split(","))
                except ValueError:
                    raise ValueError(
                        f"{key} must be a comma-separated list of numbers, got {value!r}"
                    ) from None
        return cls(**flat)


@dataclass(frozen=True)
class G1Config(_FromKwargs):
    """Static Unitree G1 hardware, safety, hand, camera, and pacing settings."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"joint_low", "joint_high", "home_pose", "rest_pose"}
    )

    iface: str | None = "eth0"
    hand_type: str = "dex3"
    arm_dof: int = 7
    control_hz: float = 10.0
    stream_hz: float = 50.0
    max_joint_speed: float = 3.0
    weight_ramp_s: float = 2.0
    # unitree_sdk2_python@e4cd91f051aa arm7 example uses 60/1.5 uniformly.
    kp_arm: float = 60.0
    kd_arm: float = 1.5
    # Bench-tuned package defaults. These are our conservative choice, not a citation.
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5
    joint_low: tuple[float, ...] = DEFAULT_JOINT_LOW
    joint_high: tuple[float, ...] = DEFAULT_JOINT_HIGH
    home_pose: tuple[float, ...] = DEFAULT_HOME_POSE
    rest_pose: tuple[float, ...] | None = None
    # xr_teleoperate@7dc9aa1a6edb robot_hand_unitree.py lines 326-333 records
    # the 5.4 rad stroke and a 0.18 rad/cycle clip at 250 Hz. Our 2.7 rad/s
    # default is a deliberately conservative package choice.
    dex1_stroke: float = 5.4
    dex1_max_speed: float = 2.7
    hand_deadband: float = 0.05
    hand_kp: float | None = None
    hand_kd: float | None = None
    # Dex3 thumb0 swing, constrained by both dex3_1 URDFs at d96d8f63ae17.
    dex3_thumb_swing: float = 0.0
    cam_server_address: str = "tcp://192.168.123.164:5556"
    cam_timeout_s: float = 5.0
    unattended: bool = False
    docs_extra: str = ""

    def __post_init__(self) -> None:
        """Reject settings that violate the fixed 16-D safety contract."""
        if self.arm_dof != 7:
            raise ConfigError("arm_dof must be 7; the 23-DOF G1 five-joint arms are out of scope")
        if self.hand_type not in {"dex1", "dex3"}:
            raise ValueError("hand_type must be 'dex1' or 'dex3'")
        for name in ("control_hz", "stream_hz", "max_joint_speed", "weight_ramp_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        ratio = self.stream_hz / self.control_hz
        if self.stream_hz < self.control_hz or not np.isclose(ratio, round(ratio)):
            raise ValueError("stream_hz must be an integer multiple of control_hz")
        for name in ("kp_arm", "kd_arm", "kp_wrist", "kd_wrist"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("joint_low", "joint_high", "home_pose"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.rest_pose is not None and len(self.rest_pose) != TOTAL_DIM:
            raise ValueError(f"rest_pose must have {TOTAL_DIM} entries")
        low, high = self.low, self.high
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("joint_low and joint_high must contain only finite values")
        if np.any(low >= high):
            raise ValueError("joint_low must be below joint_high in every dimension")
        for name in ("home_pose", "rest_pose"):
            pose = getattr(self, name)
            if pose is not None:
                values = np.asarray(pose, dtype=np.float64)
                if not np.all(np.isfinite(values)) or np.any(values < low) or np.any(values > high):
                    raise ValueError(f"{name} must be finite and inside joint_low/joint_high")
        for name in ("dex1_stroke", "dex1_max_speed", "cam_timeout_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.hand_deadband) or not 0 <= self.hand_deadband < 1:
            raise ValueError("hand_deadband must be finite and in [0, 1)")
        if not np.isfinite(self.dex3_thumb_swing) or not -1.0472 <= self.dex3_thumb_swing <= 1.0472:
            raise ValueError("dex3_thumb_swing must be finite and inside [-1.0472, 1.0472]")
        # xr_teleoperate@7dc9aa1a6edb,
        # teleop/robot_control/robot_hand_unitree.py: Dex1 5.0/0.05 and
        # Dex3 1.5/0.2.
        default_kp, default_kd = (5.0, 0.05) if self.hand_type == "dex1" else (1.5, 0.2)
        if self.hand_kp is None:
            object.__setattr__(self, "hand_kp", default_kp)
        if self.hand_kd is None:
            object.__setattr__(self, "hand_kd", default_kd)
        for name in ("hand_kp", "hand_kd"):
            value = getattr(self, name)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not self.cam_server_address:
            raise ValueError("cam_server_address must not be empty")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return configured lower action bounds."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return configured upper action bounds."""
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class Gr00tConfig(_FromKwargs):
    """Static Isaac-GR00T transport and modality-template settings."""

    host: str = "127.0.0.1"
    port: int = 5555
    timeout_s: float = 15.0
    actions_are_relative: bool = False
    action_horizon: int = 16
    replan_interval: int = 8
    name: str = "gr00t"
    image_key: str = "video.ego_view"
    hand_type: str = "dex3"
    dex1_stroke: float = 5.4
    dex3_thumb_swing: float = 0.0
    control_hz: float = 10.0
    state_keys: Mapping[str, SourceSpec] = field(default_factory=lambda: dict(DEFAULT_STATE_KEYS))
    action_keys: Mapping[str, ActionSpec] = field(default_factory=lambda: dict(DEFAULT_ACTION_KEYS))

    def __post_init__(self) -> None:
        """Validate transport values and every tagged source specification."""
        if not self.host:
            raise ValueError("host must not be empty")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer in [1, 65535]")
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        for name in ("action_horizon", "replan_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.name or "." not in self.image_key:
            raise ValueError("name must not be empty and image_key must contain a group prefix")
        if self.image_key.split(".", 1)[0] != "video":
            raise ValueError("image_key must use the 'video' group")
        if self.hand_type not in {"dex1", "dex3"}:
            raise ValueError("hand_type must be 'dex1' or 'dex3'")
        if not np.isfinite(self.dex1_stroke) or self.dex1_stroke <= 0:
            raise ValueError("dex1_stroke must be finite and > 0")
        if not np.isfinite(self.dex3_thumb_swing) or not -1.0472 <= self.dex3_thumb_swing <= 1.0472:
            raise ValueError("dex3_thumb_swing must be finite and inside [-1.0472, 1.0472]")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        for key, spec in self.state_keys.items():
            _validate_source(key, spec, action=False)
            if key.split(".", 1)[0] != "state":
                raise ValueError(f"state key {key!r} must use the 'state' group")
            if spec[0] == "state" and spec[1] not in STATE_SPEC.keys:
                raise ValueError(f"state source {spec[1]!r} is not declared by observation_space")
        for key, spec in self.action_keys.items():
            _validate_source(key, spec, action=True)
            if key.split(".", 1)[0] != "action":
                raise ValueError(f"action key {key!r} must use the 'action' group")


def _validate_source(key: str, spec: SourceSpec, *, action: bool) -> None:
    if "." not in key or not isinstance(spec, tuple) or not spec:
        raise ValueError(f"invalid source spec for {key!r}: {spec!r}")
    tag = spec[0]
    valid = {"packed", "hand"} if action else {"packed", "hand", "state", "const"}
    if tag not in valid:
        raise ValueError(f"invalid source tag {tag!r} for {key!r}")
    if tag == "packed":
        if len(spec) != 3 or not all(isinstance(value, int) for value in spec[1:]):
            raise ValueError(f"packed source for {key!r} must be ('packed', start, stop)")
        if not 0 <= spec[1] < spec[2] <= TOTAL_DIM:
            raise ValueError(f"packed source for {key!r} has invalid slice")
    elif tag == "hand":
        if len(spec) != 2 or spec[1] not in {"left", "right"}:
            raise ValueError(f"hand source for {key!r} must select left or right")
    elif tag == "state":
        if len(spec) != 2 or not isinstance(spec[1], str) or not spec[1]:
            raise ValueError(f"state source for {key!r} must name a field")
    elif len(spec) != 3 or not isinstance(spec[1], int) or spec[1] < 1 or not np.isfinite(spec[2]):
        raise ValueError(f"const source for {key!r} must be ('const', positive_dim, value)")


def action_box(cfg: G1Config | None = None) -> Box:
    """Build the shared absolute 16-D joint-position action space."""
    return Box(
        shape=(TOTAL_DIM,),
        low=cfg.low if cfg is not None else np.asarray(DEFAULT_JOINT_LOW),
        high=cfg.high if cfg is not None else np.asarray(DEFAULT_JOINT_HIGH),
        semantics=ACTION_SEMANTICS,
    )


def observation_space() -> ObservationSpace:
    """Build the one shared G1 camera and proprioception contract."""
    return ObservationSpace(
        cameras=(CameraSpec(name="head_cam", height=480, width=640, channels=3),),
        state=STATE_SPEC,
    )
