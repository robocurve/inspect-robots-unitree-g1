"""Backend seam for walking robots: the real SDK, a kinematic sim, or Isaac Lab.

Every backend speaks the same small surface: hold a body velocity, stand
still, report odometry and IMU, and disconnect. Balance and leg control stay
inside the backend (Unitree's walking controller, or a trained locomotion
policy in Isaac Lab), so the embodiment never touches leg joints.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots_unitree_g1.walking.config import (
    G1Config,
    G1MobileConfig,
    LeggedConfig,
)

Vec = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LeggedState:
    """One state sample in the backend's own odometry frame.

    ``pose`` is ``(x m, y m, yaw rad)``, ``velocity`` is the body-frame
    ``(forward m/s, left m/s, turn rad/s)``, ``rpy`` is IMU roll, pitch, yaw.
    """

    pose: Vec
    velocity: Vec
    rpy: Vec


@runtime_checkable
class LeggedDriver(Protocol):
    """The minimal backend surface the embodiment needs."""

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Command one walking velocity frame (m/s, m/s, rad/s) in the body frame."""
        ...

    def stand(self) -> None:
        """Command zero velocity and a stable standing posture."""
        ...

    def read_state(self) -> LeggedState:
        """Return the latest odometry and IMU sample."""
        ...

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Return the latest RGB camera frame as ``(H, W, 3)`` uint8."""
        ...

    def disconnect(self) -> None:
        """Release transport or simulator resources."""
        ...


DriverFactory = Callable[[LeggedConfig], LeggedDriver]


class KinematicLeggedDriver:
    """A dependency-free stand-in that integrates commanded velocity over time.

    It models the walking controller as a first-order lag toward the commanded
    velocity, integrates the body pose, and renders a top-down frame (grid,
    heading arrow) so camera-driven policies have pixels to look at. It is
    meant for mock runs and tests, not for judging gait or contact physics.
    """

    def __init__(
        self,
        cfg: LeggedConfig,
        *,
        clock: Callable[[], float],
        response_time_s: float = 0.15,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._tau = response_time_s
        self._pose = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._command = np.zeros(3, dtype=np.float64)
        self._last_time = clock()
        self._lock = threading.RLock()
        self.connected = True

    def _advance_locked(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._last_time)
        self._last_time = now
        if dt == 0.0:
            return
        alpha = 1.0 if self._tau <= 0 else 1.0 - math.exp(-dt / self._tau)
        self._velocity = self._velocity + alpha * (self._command - self._velocity)
        yaw = self._pose[2] + 0.5 * self._velocity[2] * dt
        cos, sin = math.cos(yaw), math.sin(yaw)
        self._pose = self._pose + dt * np.asarray(
            (
                cos * self._velocity[0] - sin * self._velocity[1],
                sin * self._velocity[0] + cos * self._velocity[1],
                self._velocity[2],
            )
        )

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Advance the model, then latch the new command."""
        with self._lock:
            self._advance_locked()
            self._command = np.asarray((forward, left, turn), dtype=np.float64)

    def stand(self) -> None:
        """Advance the model, then command zero velocity."""
        self.send_velocity(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """Advance the model and report pose, velocity and a level IMU."""
        with self._lock:
            self._advance_locked()
            return LeggedState(
                pose=self._pose.copy(),
                velocity=self._velocity.copy(),
                rpy=np.asarray((0.0, 0.0, self._pose[2]), dtype=np.float64),
            )

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Render a robot-centred top-down view: a 1 m grid and a heading arrow."""
        with self._lock:
            self._advance_locked()
            pose = self._pose.copy()
        return render_top_down(pose, self._cfg.camera_height, self._cfg.camera_width)

    def disconnect(self) -> None:
        """Mark the stand-in disconnected."""
        self.connected = False


def render_top_down(
    pose: Vec, height: int, width: int, px_per_m: float = 80.0
) -> npt.NDArray[np.uint8]:
    """Draw grid lines every metre in the world frame, centred on the robot.

    The robot sits at the image centre facing up; the world grid rotates and
    slides underneath it, so motion is visible between frames.
    """
    frame = np.full((height, width, 3), 235, dtype=np.uint8)
    cy, cx = height / 2.0, width / 2.0
    rows, cols = np.mgrid[0:height, 0:width]
    # Image offset from the robot in metres, robot frame: up is forward, left is left.
    fwd = (cy - rows) / px_per_m
    lft = (cx - cols) / px_per_m
    yaw = float(pose[2])
    cos, sin = math.cos(yaw), math.sin(yaw)
    world_x = pose[0] + cos * fwd - sin * lft
    world_y = pose[1] + sin * fwd + cos * lft
    tol = 1.0 / px_per_m
    grid = (np.abs(world_x - np.round(world_x)) < tol) | (np.abs(world_y - np.round(world_y)) < tol)
    frame[grid] = (150, 150, 150)
    origin = (world_x**2 + world_y**2) < (0.08**2)
    frame[origin] = (40, 120, 220)
    # Heading arrow: a short bar from the centre toward the top of the image.
    bar = (np.abs(cols - cx) < 3) & (rows < cy) & (rows > cy - 0.4 * px_per_m)
    frame[bar] = (200, 40, 40)
    return frame


def default_driver_factory(cfg: LeggedConfig) -> LeggedDriver:  # pragma: no cover - backends
    """Build the backend named by ``cfg.backend`` for the robot the config describes."""
    if cfg.backend == "kinematic":
        import time

        if isinstance(cfg, G1MobileConfig):
            from inspect_robots_unitree_g1.walking.mobile import KinematicMobileDriver

            return KinematicMobileDriver(cfg, clock=time.monotonic)
        return KinematicLeggedDriver(cfg, clock=time.monotonic)
    if cfg.backend == "isaaclab" and isinstance(cfg, G1MobileConfig):
        from inspect_robots_unitree_g1.walking._isaaclab import IsaacLabG1MobileDriver

        return IsaacLabG1MobileDriver(cfg)
    if cfg.backend == "isaaclab":
        from inspect_robots_unitree_g1.walking._isaaclab import IsaacLabVelocityDriver

        return IsaacLabVelocityDriver(cfg)
    if isinstance(cfg, G1MobileConfig):
        from inspect_robots_unitree_g1.walking.g1 import build_g1_mobile_sdk_driver

        return build_g1_mobile_sdk_driver(cfg)
    if isinstance(cfg, G1Config):
        from inspect_robots_unitree_g1.walking.g1 import build_g1_sdk_driver

        return build_g1_sdk_driver(cfg)
    raise TypeError(f"no sdk backend for {type(cfg).__name__}")
