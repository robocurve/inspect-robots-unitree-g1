"""Position-only forward and inverse kinematics for the Unitree G1 arms.

Pure NumPy. Each arm is a serial chain from ``torso_link`` to the palm point;
every joint is a fixed transform (``xyz``, ``rpy`` from the URDF) followed by
a rotation about its axis. Hand positions are expressed in the torso frame:
x forward, y left, z up, metres, origin at ``torso_link``.

IK moves only the shoulder and elbow joints (the first four) and holds the
wrist at its nominal angles: three position coordinates are reachable with
four joints, and the spare degree of freedom is spent pulling toward a
comfortable nominal posture. The solver is damped least squares, clipped to
joint limits at every iteration, so an unreachable target yields the closest
reachable pose instead of a failure.

Chain data: ``unitreerobotics/unitree_ros@725b59b``,
``robots/g1_description/g1_29dof_rev_1_0.urdf`` (left and right arms,
``torso_link`` to ``*_hand_palm_link``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Mat = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Joint:
    """One revolute joint: its fixed origin in the parent frame, axis, and limits."""

    name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class ArmChain:
    """A serial arm from the torso frame to the palm point."""

    joints: tuple[Joint, ...]
    palm: tuple[float, float, float]
    nominal: tuple[float, ...]
    #: Joints the IK may move (indices into ``joints``); the rest hold ``nominal``.
    ik_joints: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def dof(self) -> int:
        """Number of joints in the chain."""
        return len(self.joints)

    @property
    def lower(self) -> Vec:
        """Lower joint limits."""
        return np.asarray([joint.lower for joint in self.joints], dtype=np.float64)

    @property
    def upper(self) -> Vec:
        """Upper joint limits."""
        return np.asarray([joint.upper for joint in self.joints], dtype=np.float64)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> Mat:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _axis_angle(axis: Vec, angle: float) -> Mat:
    """Rodrigues rotation about a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.asarray(
        (
            (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
        ),
        dtype=np.float64,
    )


def forward(chain: ArmChain, q: npt.ArrayLike) -> tuple[Vec, list[Vec], list[Vec]]:
    """Return the palm position plus each joint's world position and world axis."""
    angles = np.asarray(q, dtype=np.float64).reshape(-1)
    if angles.shape != (chain.dof,):
        raise ValueError(f"expected {chain.dof} joint angles, got {angles.shape}")
    rotation = np.eye(3)
    position = np.zeros(3)
    origins: list[Vec] = []
    axes: list[Vec] = []
    for joint, angle in zip(chain.joints, angles, strict=True):
        position = position + rotation @ np.asarray(joint.xyz)
        rotation = rotation @ _rpy_matrix(*joint.rpy)
        axis_world = rotation @ np.asarray(joint.axis, dtype=np.float64)
        origins.append(position.copy())
        axes.append(axis_world)
        rotation = rotation @ _axis_angle(np.asarray(joint.axis, dtype=np.float64), float(angle))
    palm = position + rotation @ np.asarray(chain.palm)
    return palm, origins, axes


def fk(chain: ArmChain, q: npt.ArrayLike) -> Vec:
    """Palm position in the torso frame."""
    return forward(chain, q)[0]


def jacobian(chain: ArmChain, q: npt.ArrayLike) -> Mat:
    """Position Jacobian of the palm, shape ``(3, dof)``."""
    palm, origins, axes = forward(chain, q)
    columns = [np.cross(axis, palm - origin) for axis, origin in zip(axes, origins, strict=True)]
    return np.stack(columns, axis=1)


#: Extra IK starting postures (shoulder pitch, roll, yaw, elbow), tried when
#: the warm start leaves more than ``retry_error`` metres of error.
_SEEDS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.3, 0.0, 0.5),
    (-0.8, 0.3, 0.0, 0.8),
    (-1.5, 0.2, 0.0, 0.3),
    (0.5, 0.6, 0.5, 1.5),
)


def _solve_from(
    chain: ArmChain,
    goal: Vec,
    q: Vec,
    *,
    iterations: int,
    damping: float,
    posture_gain: float,
    tolerance: float,
    max_step: float,
) -> tuple[Vec, float]:
    lower, upper = chain.lower, chain.upper
    nominal = np.asarray(chain.nominal, dtype=np.float64)
    active = np.asarray(chain.ik_joints, dtype=np.int64)
    error = goal - fk(chain, q)
    for _ in range(iterations):
        if float(np.linalg.norm(error)) < tolerance:
            break
        jac = jacobian(chain, q)[:, active]
        damped = jac @ jac.T + (damping**2) * np.eye(3)
        step = jac.T @ np.linalg.solve(damped, error)
        # Spend the spare degree of freedom pulling toward the nominal posture.
        null = np.eye(len(active)) - np.linalg.pinv(jac) @ jac
        step = step + posture_gain * null @ (nominal[active] - q[active])
        largest = float(np.max(np.abs(step)))
        if largest > max_step:
            step = step * (max_step / largest)
        q[active] = np.clip(q[active] + step, lower[active], upper[active])
        error = goal - fk(chain, q)
    return q, float(np.linalg.norm(error))


def ik(
    chain: ArmChain,
    target: npt.ArrayLike,
    q_start: npt.ArrayLike,
    *,
    iterations: int = 40,
    damping: float = 0.02,
    posture_gain: float = 0.05,
    tolerance: float = 1e-4,
    max_step: float = 0.2,
    retry_error: float = 5e-3,
) -> tuple[Vec, float]:
    """Solve for joint angles that put the palm at ``target``.

    Starts from ``q_start`` (warm start); if that leaves more than
    ``retry_error`` metres, retries from a few fixed postures and keeps the
    best. Returns the solution (always within joint limits, wrist at nominal)
    and its remaining position error in metres.
    """
    goal = np.asarray(target, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(goal)):
        raise ValueError("IK target must be finite")
    lower, upper = chain.lower, chain.upper
    nominal = np.asarray(chain.nominal, dtype=np.float64)
    passive = np.setdiff1d(np.arange(chain.dof), np.asarray(chain.ik_joints))
    starts = [np.asarray(q_start, dtype=np.float64).reshape(-1)]
    starts += [np.asarray((*seed, *chain.nominal[4:]), dtype=np.float64) for seed in _SEEDS]
    best: tuple[Vec, float] | None = None
    for index, start in enumerate(starts):
        q = np.clip(start, lower, upper)
        q[passive] = nominal[passive]
        if index > 0:
            # Seeds are written for the left arm; mirror roll and yaw for the right.
            q[1:3] *= np.sign(chain.nominal[1])
            q = np.clip(q, lower, upper)
        result = _solve_from(
            chain,
            goal,
            q,
            iterations=iterations,
            damping=damping,
            posture_gain=posture_gain,
            tolerance=tolerance,
            max_step=max_step,
        )
        if best is None or result[1] < best[1]:
            best = result
        if best[1] <= retry_error:
            break
    assert best is not None
    return best


def _arm(side: str) -> ArmChain:
    """The G1 29-DoF rev 1.0 arm. The right arm mirrors y and the roll angles."""
    m = 1.0 if side == "left" else -1.0
    roll_limits = (-1.5882, 2.2515) if side == "left" else (-2.2515, 1.5882)
    joints = (
        Joint(
            f"{side}_shoulder_pitch_joint",
            (0.0039563, m * 0.10022 if side == "left" else -0.10021, 0.24778),
            (m * 0.27931, 5.4949e-05, m * -0.00019159),
            (0.0, 1.0, 0.0),
            -3.0892,
            2.6704,
        ),
        Joint(
            f"{side}_shoulder_roll_joint",
            (0.0, m * 0.038, -0.013831),
            (m * -0.27925, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            *roll_limits,
        ),
        Joint(
            f"{side}_shoulder_yaw_joint",
            (0.0, m * 0.00624, -0.1032),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            -2.618,
            2.618,
        ),
        Joint(
            f"{side}_elbow_joint",
            (0.015783, 0.0, -0.080518),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            -1.0472,
            2.0944,
        ),
        Joint(
            f"{side}_wrist_roll_joint",
            (0.100, m * 0.00188791, -0.010),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            -1.972222054,
            1.972222054,
        ),
        Joint(
            f"{side}_wrist_pitch_joint",
            (0.038, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            -1.614429558,
            1.614429558,
        ),
        Joint(
            f"{side}_wrist_yaw_joint",
            (0.046, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            -1.614429558,
            1.614429558,
        ),
    )
    # Nominal: upper arm slightly forward and out, elbow bent, wrist straight.
    nominal = (0.3, m * 0.2, 0.0, 1.0, 0.0, 0.0, 0.0)
    return ArmChain(joints=joints, palm=(0.0415, m * 0.003, 0.0), nominal=nominal)


G1_LEFT_ARM = _arm("left")
G1_RIGHT_ARM = _arm("right")
G1_ARMS: tuple[ArmChain, ArmChain] = (G1_LEFT_ARM, G1_RIGHT_ARM)


def with_tool_offset(arms: tuple[ArmChain, ArmChain], offset_m: float) -> tuple[ArmChain, ArmChain]:
    """Move each arm's controlled point ``offset_m`` further along the hand's x axis.

    With the wrist at its nominal (straight) angles the hand's x axis points
    along the forearm, so this tracks a gripper's grasp point instead of the palm.
    """
    if offset_m == 0.0:
        return arms
    left, right = (
        replace(arm, palm=(arm.palm[0] + offset_m, arm.palm[1], arm.palm[2])) for arm in arms
    )
    return left, right
