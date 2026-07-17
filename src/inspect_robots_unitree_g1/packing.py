"""Pure helpers for the canonical 16-D Unitree G1 arm and hand packing.

The vector is blockwise left then right. Each block contains seven absolute
arm joint angles followed by one normalized hand value, where 1 means open.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ARM_DOF = 7
ARM_WIDTH = 8
TOTAL_DIM = 16
LEFT = slice(0, ARM_WIDTH)
RIGHT = slice(ARM_WIDTH, TOTAL_DIM)
GRIPPER_IDXS = (7, 15)
STATE_KEY = "joint_pos"

DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(1, ARM_DOF + 1)), "gripper")
)

# unitreerobotics/unitree_ros@d96d8f63ae17,
# robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf. xr_teleoperate
# has no fixed poses, so these conservative interior poses are the documented
# URDF-derived fallback. Motor order is from xr_teleoperate@7dc9aa1a6edb,
# teleop/robot_control/robot_hand_unitree.py lines 213-229.
DEX3_OPEN_POSE: tuple[float, ...] = (0.0,) * 7
DEX3_LEFT_CLOSED_POSE: tuple[float, ...] = (0.0, 0.6, 0.8, -1.2, -1.4, -1.2, -1.4)
DEX3_RIGHT_CLOSED_POSE: tuple[float, ...] = (0.0, -0.6, -0.8, 1.2, 1.4, 1.2, 1.4)

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return a one-dimensional float64 vector of exactly ``n`` elements."""
    arr: Vec = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"expected a {n}-D vector, got shape {np.shape(vec)}")
    return arr


def pack(left: npt.ArrayLike, right: npt.ArrayLike) -> Vec:
    """Pack two eight-wide arm-plus-hand blocks into the canonical vector."""
    return np.concatenate((validate_dim(left, ARM_WIDTH), validate_dim(right, ARM_WIDTH)))


def split(vec: npt.ArrayLike) -> tuple[Vec, Vec]:
    """Split a canonical vector into independent left and right blocks."""
    arr = validate_dim(vec)
    return arr[LEFT].copy(), arr[RIGHT].copy()


def arm_slots(vec: npt.ArrayLike) -> Vec:
    """Return the 14 arm joints in Unitree SDK order, left then right."""
    arr = validate_dim(vec)
    return np.concatenate((arr[:ARM_DOF], arr[ARM_WIDTH : ARM_WIDTH + ARM_DOF]))


def dex1_scalar_to_stroke(wire: float, stroke: float = 5.4) -> float:
    """Map normalized open-positive closure to the Dex1 motor stroke.

    xr_teleoperate@7dc9aa1a6edb, robot_hand_unitree.py lines 326-333 records
    0 rad as closed and 5.4 rad as open.
    """
    return float(np.clip(wire, 0.0, 1.0) * stroke)


def dex1_stroke_to_scalar(value: float, stroke: float = 5.4) -> float:
    """Map a Dex1 motor stroke back to normalized open-positive closure."""
    if not np.isfinite(stroke) or stroke <= 0:
        raise ValueError("stroke must be finite and > 0")
    return float(np.clip(value / stroke, 0.0, 1.0))


def dex3_scalar_to_joints(wire: float, open_pose: npt.ArrayLike, closed_pose: npt.ArrayLike) -> Vec:
    """Interpolate a seven-joint hand pose, with 1 open and 0 closed."""
    opened = validate_dim(open_pose, 7)
    closed = validate_dim(closed_pose, 7)
    alpha = float(np.clip(wire, 0.0, 1.0))
    return closed + alpha * (opened - closed)


def dex3_joints_to_scalar(
    joints: npt.ArrayLike, open_pose: npt.ArrayLike, closed_pose: npt.ArrayLike
) -> float:
    """Compress a seven-joint hand pose to mean normalized openness."""
    values = validate_dim(joints, 7)
    opened = validate_dim(open_pose, 7)
    closed = validate_dim(closed_pose, 7)
    span = opened - closed
    moving = np.abs(span) > np.finfo(np.float64).eps
    if not np.any(moving):
        raise ValueError("open_pose and closed_pose must differ in at least one joint")
    normalized = (values[moving] - closed[moving]) / span[moving]
    return float(np.clip(np.mean(normalized), 0.0, 1.0))
