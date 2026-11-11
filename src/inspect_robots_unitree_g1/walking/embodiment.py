"""Walking embodiments: steer a body by small, watched displacements.

Each action is one control period's body displacement ``(forward m, left m,
turn rad)``. ``step()`` clamps it, converts it to a body velocity, hands that
to the command streamer with a deadline, waits one control period, and
observes. The streamer re-sends the current command at ``stream_hz`` and
drops to standing once the deadline passes, so the robot stops by itself
whenever steps stop arriving (a slow policy, a crash, a closed run).
"""

from __future__ import annotations

import atexit
import contextlib
import math
import signal
import threading
import time
from collections.abc import Callable, Mapping
from types import FrameType
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import (
    SELF_PACED,
    EmbodimentInfo,
    HealthCheck,
    HealthReport,
)
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_unitree_g1._unitree import UNITREE_INSTALL_COMMAND
from inspect_robots_unitree_g1.operator import OperatorIO, default_poll_end
from inspect_robots_unitree_g1.walking.config import (
    ACTION_DIM,
    POSE_KEY,
    RPY_KEY,
    VEL_KEY,
    G1Config,
    LeggedConfig,
)
from inspect_robots_unitree_g1.walking.driver import (
    DriverFactory,
    LeggedDriver,
    LeggedState,
    default_driver_factory,
)

Vec = npt.NDArray[np.float64]
CameraReader = Callable[[], npt.NDArray[np.uint8]]
ThreadFactory = Callable[[Callable[[], None]], Any]

_DOCS = """A robot walking on flat ground. Its own walking controller handles
the legs and balance; you only steer the body.
Each action is how far the body should move during one control step, in the
robot's own frame at the start of that step:
- base_forward: metres toward the front of the body (negative walks backward).
- base_left: metres toward the robot's left side (negative steps right).
- base_turn: radians of turn, positive counter-clockwise seen from above.
A move can be larger than one step's bound; it is split into steps at full
speed. Prefer moves under about 1 m or 90 degrees, then look again.
The robot stops on its own a moment after commands stop. base_pose in the
state is odometry (x, y, yaw) from where it stood at reset; it drifts over
a few metres, so trust the camera for where things are. Move in short
segments and look again before the next one."""


_G1_DOCS = """A Unitree G1 humanoid walking on flat ground. Its own locomotion
controller handles the legs and balance; you only steer the body.
Each action is how far the body should move during one control step, in the
robot's own frame at the start of that step:
- base_forward: metres toward the front of the torso (negative walks backward).
- base_left: metres toward the robot's left side (negative steps right).
- base_turn: radians of turn, positive counter-clockwise seen from above.
A move can be larger than one step's bound; it is split into steps at full
speed. Prefer moves under about 1 m or 90 degrees, then look again. Side
steps are slow for a humanoid: to go somewhere to the side, turn to face it
and walk forward.
The robot stops on its own a moment after commands stop, and it needs a
moment to settle after stopping. base_pose in the state is odometry
(x, y, yaw) from where it stood at reset; it drifts, so trust the head camera
for where things are. Move in short segments and look again before the next
one."""


def _thread_factory(target: Callable[[], None]) -> threading.Thread:  # pragma: no cover
    thread = threading.Thread(target=target, name="walking-command-streamer", daemon=True)
    thread.start()
    return thread


class LeggedEmbodiment:
    """Shared base: a walking body steered through planar velocity.

    Other walking robots subclass it and override the class attributes below;
    the stepping, streaming, watchdog, geofence and teardown logic is shared.
    """

    NAME: ClassVar[str] = "legged_base"
    LABEL: ClassVar[str] = "Robot"
    MORPHOLOGY: ClassVar[str] = "legged"
    ENV_PREFIX: ClassVar[str] = "legged"
    CONFIG_CLS: ClassVar[type[LeggedConfig]] = LeggedConfig
    DOCS: ClassVar[str] = _DOCS
    READY_PROMPT: ClassVar[str] = (
        "The robot will stand and accept walking commands. Clear the area, then press Enter..."
    )

    # Empty on purpose: each backend has different needs (none for kinematic,
    # the robot SDK for sdk, Isaac Lab for isaaclab), and doctor checks this
    # without knowing the backend. Backends raise their own install hint.
    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {}

    def __init__(
        self,
        config: LeggedConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        thread_factory: ThreadFactory | None = None,
        atexit_module: Any = atexit,
        signal_module: Any = signal,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else self.CONFIG_CLS.from_kwargs(**flat)
        if not isinstance(self._cfg, self.CONFIG_CLS):
            raise TypeError(
                f"{type(self).__name__} needs a {self.CONFIG_CLS.__name__}; "
                f"got {type(self._cfg).__name__}"
            )
        self._driver_factory = driver_factory or default_driver_factory
        self._camera_reader = camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end = poll_end or default_poll_end
        self._clock = clock or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self._wall_clock, self._wall_sleep = self._clock, self._sleep
        self._sim_paced = False
        self._thread_factory = thread_factory or _thread_factory
        self._atexit = atexit_module
        self._signal = signal_module
        self._driver: LeggedDriver | None = None
        self._lock = threading.Lock()
        self._command = np.zeros(ACTION_DIM, dtype=np.float64)
        self._deadline = -math.inf
        self._standing = True
        self._stop_streamer = threading.Event()
        self._streamer: Any = None
        self._stream_error: BaseException | None = None
        self._origin: LeggedState | None = None
        self._instruction: str | None = None
        self._backstops_registered = False
        self._previous_sigterm: Any = None
        self.num_steps = 0
        docs = self.DOCS
        if self._cfg.docs_extra.strip():
            docs += "\n\n" + self._cfg.docs_extra.strip()
        self.info = EmbodimentInfo(
            name=self.NAME,
            action_space=self._cfg.make_action_box(),
            observation_space=self._cfg.make_observation_space(),
            control_hz=self._cfg.control_hz,
            is_simulated=self._cfg.backend != "sdk",
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
            morphology=self.MORPHOLOGY,
            environment_id=f"{self.ENV_PREFIX}:{self._cfg.backend}",
        )

    # ------------------------------------------------------------------ lifecycle

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, stand, wait for the operator, and zero odometry here."""
        del seed
        if self._camera_reader is not None and not callable(self._camera_reader):
            raise ValueError("camera_reader must be callable when injected")
        if self._driver is None:
            self._connect()
        driver = self._require_driver()
        self._hold_stand()
        driver.stand()
        if not self._cfg.unattended:
            self._operator.wait_ready(self.READY_PROMPT)
        try:
            self._reset_extra(driver)
        except BaseException:
            self.close()
            raise
        self._origin = self._read_state(driver)
        self._instruction = scene.instruction
        self.num_steps = 0
        try:
            return self._observe()
        except BaseException:
            self.close()
            raise

    def step(self, action: Action) -> StepResult:
        """Clamp, command one control period of walking, pace, observe, and fence."""
        self._require_driver()
        self._raise_stream_error()
        command = np.asarray(action.data, dtype=np.float64).reshape(-1)
        low, high = self._cfg.low, self._cfg.high
        if command.shape != low.shape:
            raise ValueError(
                f"{self.LABEL} action must have {low.size} values; got {command.shape}"
            )
        if not np.isfinite(command).all():
            raise ValueError("action must contain only finite values")
        clamped = np.clip(command, low, high)
        velocity = clamped[:ACTION_DIM] * self._cfg.control_hz
        period = 1.0 / self._cfg.control_hz
        try:
            with self._lock:
                self._command = velocity
                self._standing = False
                self._deadline = self._clock() + period + self._cfg.command_timeout_s
                self._apply_extra_locked(clamped[ACTION_DIM:])
            if self._sim_paced:
                # Latch the command before physics runs, not on the streamer's next tick.
                self.stream_once()
            self._sleep(period)
            self.num_steps += 1
            observation = self._observe()
        except BaseException:
            self.close()
            raise
        self._raise_stream_error()
        distance = float(np.hypot(*observation.state[POSE_KEY][:2]))
        if self._cfg.geofence_m is not None and distance > self._cfg.geofence_m:
            self._hold_stand()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="geofence",
                info={"distance_m": distance, "geofence_m": self._cfg.geofence_m},
            )
        if not self._cfg.unattended and self._poll_end():
            self._hold_stand()
            success = self._operator.confirm_success()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=observation, terminated=False)

    def health(self) -> HealthReport:
        """Report a connected backend's state as finite and the streamer as alive."""
        driver = self._driver
        if driver is None:
            return HealthReport()
        checks: list[HealthCheck] = []
        try:
            self._read_state(driver)
            checks.append(HealthCheck("state", True))
        except Exception as exc:
            checks.append(HealthCheck("state", False, str(exc)))
        error = self._stream_error
        checks.append(
            HealthCheck(
                "streamer",
                error is None,
                "" if error is None else f"command streamer failed: {error}",
            )
        )
        return HealthReport(tuple(checks))

    def park(self) -> None:
        """Stop walking and stand in place. Never releases the backend."""
        if self._driver is None:
            return
        self._hold_stand()
        self._driver.stand()

    def close(self) -> None:
        """Stand, stop the streamer, and always disconnect the backend."""
        driver = self._driver
        if driver is None:
            return
        error: BaseException | None = None
        self._hold_stand()
        self._stop_streamer.set()
        streamer, self._streamer = self._streamer, None
        if streamer is not None and hasattr(streamer, "join"):
            try:
                streamer.join(timeout=1.0)
            except BaseException as exc:  # pragma: no cover - join on a live thread
                error = exc
        try:
            self._release_extra(driver)
        except BaseException as exc:
            error = error or exc
        try:
            driver.stand()
        except BaseException as exc:
            error = error or exc
        try:
            driver.disconnect()
        except BaseException as exc:
            error = error or exc
        self._driver = None
        self._origin = None
        self._clock, self._sleep = self._wall_clock, self._wall_sleep
        self._sim_paced = False
        if error is None:
            self._unregister_backstops()
            return
        raise error

    def __del__(self) -> None:
        """Best-effort final stop when normal teardown was skipped."""
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------ streamer

    def stream_once(self) -> None:
        """Send the current command, or stand once its deadline has passed.

        The streamer thread calls this at ``stream_hz``; tests call it directly.
        """
        driver = self._driver
        if driver is None:
            return
        with self._lock:
            expired = self._clock() > self._deadline
            if expired:
                self._command = np.zeros(ACTION_DIM, dtype=np.float64)
            velocity = self._command.copy()
            standing = self._standing or expired
            self._standing = standing
        if standing:
            driver.stand()
        else:
            driver.send_velocity(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        self._stream_extra(driver)

    # ------------------------------------------------------------------ extension hooks
    # Subclasses that drive more than the base (arms, hands) override these. The
    # base embodiment has nothing extra: every hook is a no-op.

    def _reset_extra(self, driver: LeggedDriver) -> None:
        """Bring extra effectors to their start pose after the robot stands."""

    def _apply_extra_locked(self, extra: Vec) -> None:
        """Take the clamped action values after the base three (lock held)."""

    def _stream_extra(self, driver: LeggedDriver) -> None:
        """Send extra effector commands on every streamer tick."""

    def _extra_state(self, driver: LeggedDriver) -> dict[str, Vec]:
        """Extra proprioception for the observation."""
        return {}

    def _release_extra(self, driver: LeggedDriver) -> None:
        """Hand extra effectors back safely during close, after the streamer stops."""

    def _stream_loop(self) -> None:
        period = 1.0 / self._cfg.stream_hz
        while not self._stop_streamer.is_set():
            try:
                self.stream_once()
            except BaseException as exc:
                self._stream_error = exc
                return
            self._stop_streamer.wait(period)

    def _hold_stand(self) -> None:
        with self._lock:
            self._command = np.zeros(ACTION_DIM, dtype=np.float64)
            self._standing = True
            self._deadline = -math.inf

    def _raise_stream_error(self) -> None:
        error = self._stream_error
        if error is not None:
            self.close()
            raise RuntimeError(f"{self.LABEL} command streamer stopped: {error}") from error

    # ------------------------------------------------------------------ helpers

    def _connect(self) -> None:
        driver = self._driver_factory(self._cfg)
        self._driver = driver
        sim_time = getattr(driver, "sim_time", None)
        advance = getattr(driver, "advance", None)
        if callable(sim_time) and callable(advance):
            # A simulator with its own clock: pace steps and deadlines by sim time.
            self._clock, self._sleep = sim_time, advance
            self._sim_paced = True
        self._stream_error = None
        self._stop_streamer = threading.Event()
        try:
            self._register_backstops()
            self._streamer = self._thread_factory(self._stream_loop)
        except BaseException:
            self._unregister_backstops()
            self._driver = None
            self._clock, self._sleep = self._wall_clock, self._wall_sleep
            self._sim_paced = False
            driver.disconnect()
            raise

    def _register_backstops(self) -> None:
        if self._backstops_registered:
            return
        self._atexit.register(self.close)
        try:
            self._previous_sigterm = self._signal.getsignal(self._signal.SIGTERM)
            self._signal.signal(self._signal.SIGTERM, self._handle_sigterm)
        except BaseException:
            self._atexit.unregister(self.close)
            raise
        self._backstops_registered = True

    def _unregister_backstops(self) -> None:
        if not self._backstops_registered:
            return
        self._atexit.unregister(self.close)
        self._signal.signal(self._signal.SIGTERM, self._previous_sigterm)
        self._backstops_registered = False
        self._previous_sigterm = None

    def _handle_sigterm(self, signum: int, frame: FrameType | None) -> None:
        previous = self._previous_sigterm
        try:
            self.close()
        finally:
            if callable(previous):
                previous(signum, frame)

    def _require_driver(self) -> LeggedDriver:
        if self._driver is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._driver

    @staticmethod
    def _read_state(driver: LeggedDriver) -> LeggedState:
        state = driver.read_state()
        for name in ("pose", "velocity", "rpy"):
            values = np.asarray(getattr(state, name), dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"backend returned invalid {name}: {values!r}")
        return state

    def _relative_pose(self, state: LeggedState) -> Vec:
        origin = self._origin if self._origin is not None else state
        dx, dy = float(state.pose[0] - origin.pose[0]), float(state.pose[1] - origin.pose[1])
        yaw0 = float(origin.pose[2])
        cos, sin = math.cos(-yaw0), math.sin(-yaw0)
        dyaw = math.atan2(
            math.sin(float(state.pose[2]) - yaw0), math.cos(float(state.pose[2]) - yaw0)
        )
        return np.asarray((cos * dx - sin * dy, sin * dx + cos * dy, dyaw), dtype=np.float64)

    def _observe(self) -> Observation:
        driver = self._require_driver()
        state = self._read_state(driver)
        reader = self._camera_reader or driver.read_frame
        frame = np.asarray(reader(), dtype=np.uint8)
        expected = (self._cfg.camera_height, self._cfg.camera_width, 3)
        if frame.shape != expected:
            raise ValueError(f"camera frame shape {frame.shape} != declared {expected}")
        observed_at = self._clock()
        name = self._cfg.camera_name
        return Observation(
            images={name: frame},
            state={
                POSE_KEY: self._relative_pose(state),
                VEL_KEY: np.asarray(state.velocity, dtype=np.float64).copy(),
                RPY_KEY: np.asarray(state.rpy, dtype=np.float64).copy(),
                **self._extra_state(driver),
            },
            instruction=self._instruction,
            image_times={name: observed_at},
            state_time=observed_at,
        )


class G1Embodiment(LeggedEmbodiment):
    """A Unitree G1 humanoid walked through its built-in locomotion controller."""

    NAME: ClassVar[str] = "g1_base"
    LABEL: ClassVar[str] = "G1"
    MORPHOLOGY: ClassVar[str] = "unitree-g1"
    ENV_PREFIX: ClassVar[str] = "g1"
    CONFIG_CLS: ClassVar[type[LeggedConfig]] = G1Config
    DOCS: ClassVar[str] = _G1_DOCS
    READY_PROMPT: ClassVar[str] = (
        "The G1 must already be standing in walking mode (set with the Unitree remote). "
        "It will accept walking commands. Clear the area, keep the remote in hand and a "
        "spotter or gantry ready, then press Enter..."
    )
    # The same install hints as g1_arms: the real robot needs the Unitree SDK,
    # and the head camera arrives as JPEG over ZMQ. (Simulator and kinematic
    # backends need none of these; doctor reports what the real robot needs.)
    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "unitree_sdk2py": UNITREE_INSTALL_COMMAND,
        "zmq": "pip install pyzmq",
        "cv2": "pip install opencv-python-headless",
    }
    #: The robot is reached by network interface, which no device probe covers.
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = ()
