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
    Go1Config,
    LeggedConfig,
)

Vec = npt.NDArray[np.float64]

ROBOT_INTERFACE_INSTALL = (
    "Build unitree_legged_sdk v3.8 with Python bindings "
    "(https://github.com/unitreerobotics/unitree_legged_sdk, branch v3.8.0, "
    "cmake -DPYTHON_BUILD=TRUE) and put the built robot_interface module on PYTHONPATH"
)


@dataclass(frozen=True)
class Go1State:
    """One state sample in the backend's own odometry frame.

    ``pose`` is ``(x m, y m, yaw rad)``, ``velocity`` is the body-frame
    ``(forward m/s, left m/s, turn rad/s)``, ``rpy`` is IMU roll, pitch, yaw.
    """

    pose: Vec
    velocity: Vec
    rpy: Vec


@runtime_checkable
class Go1Driver(Protocol):
    """The minimal backend surface the embodiment needs."""

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Command one walking velocity frame (m/s, m/s, rad/s) in the body frame."""
        ...

    def stand(self) -> None:
        """Command zero velocity and a stable standing posture."""
        ...

    def read_state(self) -> Go1State:
        """Return the latest odometry and IMU sample."""
        ...

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Return the latest RGB camera frame as ``(H, W, 3)`` uint8."""
        ...

    def disconnect(self) -> None:
        """Release transport or simulator resources."""
        ...


DriverFactory = Callable[[LeggedConfig], Go1Driver]


class KinematicGo1Driver:
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

    def read_state(self) -> Go1State:
        """Advance the model and report pose, velocity and a level IMU."""
        with self._lock:
            self._advance_locked()
            return Go1State(
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


def default_driver_factory(cfg: LeggedConfig) -> Go1Driver:  # pragma: no cover - backends
    """Build the backend named by ``cfg.backend`` for the robot the config describes."""
    if cfg.backend == "kinematic":
        import time

        if isinstance(cfg, G1MobileConfig):
            from inspect_robots_unitree_g1.walking.mobile import KinematicMobileDriver

            return KinematicMobileDriver(cfg, clock=time.monotonic)
        return KinematicGo1Driver(cfg, clock=time.monotonic)
    if cfg.backend == "isaaclab" and isinstance(cfg, G1MobileConfig):
        from inspect_robots_unitree_g1.walking._isaaclab import IsaacLabG1MobileDriver

        return IsaacLabG1MobileDriver(cfg)
    if cfg.backend == "isaaclab":
        from inspect_robots_unitree_g1.walking._isaaclab import IsaacLabGo1Driver

        return IsaacLabGo1Driver(cfg)
    if isinstance(cfg, G1MobileConfig):
        from inspect_robots_unitree_g1.walking.g1 import build_g1_mobile_sdk_driver

        return build_g1_mobile_sdk_driver(cfg)
    if isinstance(cfg, G1Config):
        from inspect_robots_unitree_g1.walking.g1 import build_g1_sdk_driver

        return build_g1_sdk_driver(cfg)
    if isinstance(cfg, Go1Config):
        return _sdk_driver(cfg)
    raise TypeError(f"no sdk backend for {type(cfg).__name__}")


def _sdk_driver(cfg: Go1Config) -> Go1Driver:  # pragma: no cover - hardware
    """Open the legacy high-level UDP link (unitree_legged_sdk v3.8 robot_interface).

    Not yet verified on a physical Go1. Mode 2 is walking with velocity
    commands and mode 1 is forced stand, per the SDK's example_walk.py.
    Camera frames come from the Go1's head camera service, which this
    backend does not wrap yet; inject ``camera_reader`` on the embodiment.
    """
    try:
        import robot_interface as sdk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(ROBOT_INTERFACE_INSTALL, name=exc.name) from exc

    highlevel = 0xEE
    udp = sdk.UDP(highlevel, cfg.local_port, cfg.robot_ip, cfg.robot_port)
    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    class _SdkDriver:
        def _send(self, mode: int, forward: float, left: float, turn: float) -> None:
            udp.Recv()
            udp.GetRecv(state)
            cmd.mode = mode
            cmd.gaitType = 1 if mode == 2 else 0
            cmd.velocity = [float(forward), float(left)]
            cmd.yawSpeed = float(turn)
            cmd.bodyHeight = 0.0
            cmd.euler = [0.0, 0.0, 0.0]
            udp.SetSend(cmd)
            udp.Send()

        def send_velocity(self, forward: float, left: float, turn: float) -> None:
            self._send(2, forward, left, turn)

        def stand(self) -> None:
            self._send(1, 0.0, 0.0, 0.0)

        def read_state(self) -> Go1State:
            udp.Recv()
            udp.GetRecv(state)
            yaw = float(state.imu.rpy[2])
            return Go1State(
                pose=np.asarray((state.position[0], state.position[1], yaw), dtype=np.float64),
                velocity=np.asarray(
                    (state.velocity[0], state.velocity[1], state.yawSpeed), dtype=np.float64
                ),
                rpy=np.asarray(state.imu.rpy, dtype=np.float64),
            )

        def read_frame(self) -> npt.NDArray[np.uint8]:
            raise RuntimeError(
                "the sdk backend has no built-in camera; inject camera_reader on Go1Embodiment"
            )

        def disconnect(self) -> None:
            self.stand()

    return _SdkDriver()
