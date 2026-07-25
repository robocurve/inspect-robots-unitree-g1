"""Safety-choreographed Unitree G1 dual-arm embodiment.

Commands are synchronous and streamed without background threads. Every target
is hard-clamped, speed-limited, and published with the arm-sdk blend weight.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import signal
import time
from collections.abc import Callable, Mapping
from types import FrameType
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_unitree_g1 import packing
from inspect_robots_unitree_g1._unitree import UNITREE_INSTALL_COMMAND, _load_unitree
from inspect_robots_unitree_g1.config import G1Config, action_box, observation_space
from inspect_robots_unitree_g1.operator import OperatorIO, default_poll_end

Vec = npt.NDArray[np.float64]
CameraReader = Callable[[], npt.NDArray[np.uint8]]

_DOCS = """A standing Unitree G1 with two seven-joint arms and scalar hands.
Actions are absolute arm radians in the base frame. Each hand is normalized,
with 0 closed and 1 open.
- left_j1: left shoulder pitch.
- left_j2: left shoulder roll.
- left_j3: left shoulder yaw.
- left_j4: left elbow pitch.
- left_j5: left wrist roll.
- left_j6: left wrist pitch.
- left_j7: left wrist yaw.
- left_gripper: normalized left-hand opening, 0 closed and 1 open.
- right_j1: right shoulder pitch.
- right_j2: right shoulder roll.
- right_j3: right shoulder yaw.
- right_j4: right elbow pitch.
- right_j5: right wrist roll.
- right_j6: right wrist pitch.
- right_j7: right wrist yaw.
- right_gripper: normalized right-hand opening, 0 closed and 1 open.
The locomotion controller owns the legs and waist. Keep the remote operator at
the damping control and use small, observed changes."""


@runtime_checkable
class TaskEnvelopeLike(Protocol):
    """Read-only rollout metadata accepted by the task-binding hook."""

    @property
    def max_steps(self) -> int:
        """Return the framework-enforced rollout horizon."""
        ...


@runtime_checkable
class ArmDriver(Protocol):
    """Minimal arm-sdk surface needed by the embodiment."""

    def read_arm_joints(self) -> npt.NDArray[np.floating[Any]]:
        """Read the 14 arm joints in SDK order."""
        ...

    def read_body_state(self) -> Mapping[str, npt.NDArray[np.floating[Any]]]:
        """Read the left_leg (6,), right_leg (6,), and waist (3,) joints."""
        ...

    def publish_arm(
        self,
        q14: npt.NDArray[np.floating[Any]],
        weight: float,
        kp: npt.NDArray[np.floating[Any]],
        kd: npt.NDArray[np.floating[Any]],
    ) -> None:
        """Publish arm targets, gains, and blend weight in one command."""
        ...

    def disconnect(self) -> None:
        """Release DDS resources."""
        ...


@runtime_checkable
class HandDriver(Protocol):
    """Normalized two-hand driver surface needed by the embodiment."""

    def read_closure(self) -> npt.NDArray[np.floating[Any]]:
        """Read normalized open-positive left and right hand values."""
        ...

    def publish_closure(self, left: float, right: float) -> None:
        """Publish normalized open-positive hand targets."""
        ...

    def disconnect(self) -> None:
        """Release hand DDS resources."""
        ...


ArmDriverFactory = Callable[[G1Config], ArmDriver]
HandDriverFactory = Callable[[G1Config], HandDriver]


_DDS_INITIALIZED: bool = False


def _init_dds(iface: str | None, channel_factory_init_fn: Any) -> None:
    global _DDS_INITIALIZED
    if not _DDS_INITIALIZED:
        if iface is None:
            channel_factory_init_fn(0)
        else:
            channel_factory_init_fn(0, iface)
        _DDS_INITIALIZED = True


def _default_arm_driver_factory(cfg: G1Config) -> ArmDriver:  # pragma: no cover - hardware
    """Construct the real rt/arm_sdk publisher and rt/lowstate subscriber."""
    _load_unitree()
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    _init_dds(cfg.iface, ChannelFactoryInitialize)
    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    latest: list[Any] = [None]
    subscriber.Init(lambda msg: latest.__setitem__(0, msg), 10)
    command = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    # unitree_sdk2_python@e4cd91f051aa,
    # example/g1/high_level/g1_arm7_sdk_dds_example.py: arms 15-28,
    # legs 0-11, waist 12-14, and weight motor_cmd[29].q.
    arm_indices = tuple(range(15, 29))

    def state() -> Any:
        if latest[0] is None:
            raise RuntimeError("rt/lowstate has not produced a sample yet")
        return latest[0]

    class _RealArmDriver:
        def read_arm_joints(self) -> npt.NDArray[np.float64]:
            return np.asarray([state().motor_state[index].q for index in arm_indices])

        def read_body_state(self) -> Mapping[str, npt.NDArray[np.float64]]:
            values = np.asarray([state().motor_state[index].q for index in range(15)])
            return {"left_leg": values[:6], "right_leg": values[6:12], "waist": values[12:15]}

        def publish_arm(
            self,
            q14: npt.NDArray[np.floating[Any]],
            weight: float,
            kp: npt.NDArray[np.floating[Any]],
            kd: npt.NDArray[np.floating[Any]],
        ) -> None:
            for slot, index in enumerate(arm_indices):
                motor = command.motor_cmd[index]
                motor.q = float(q14[slot])
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = float(kp[slot])
                motor.kd = float(kd[slot])
            command.motor_cmd[29].q = float(weight)
            command.crc = crc.Crc(command)
            publisher.Write(command)

        def disconnect(self) -> None:
            return None

    return _RealArmDriver()


def _default_hand_driver_factory(cfg: G1Config) -> HandDriver:  # pragma: no cover - hardware
    """Construct either the real Dex1 or Dex3 DDS driver."""
    _load_unitree()
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

    if cfg.hand_type == "dex1":
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        topics = ("rt/dex1/left", "rt/dex1/right")
        publishers = [ChannelPublisher(f"{topic}/cmd", MotorCmds_) for topic in topics]
        subscribers = [ChannelSubscriber(f"{topic}/state", MotorStates_) for topic in topics]
        commands = [MotorCmds_(), MotorCmds_()]
        latest: list[Any] = [None, None]
        for index, (publisher, subscriber, command) in enumerate(
            zip(publishers, subscribers, commands, strict=True)
        ):
            publisher.Init()
            command.cmds = [unitree_go_msg_dds__MotorCmd_()]
            command.cmds[0].kp = cast(float, cfg.hand_kp)
            command.cmds[0].kd = cast(float, cfg.hand_kd)
            subscriber.Init(lambda msg, i=index: latest.__setitem__(i, msg), 10)

        class _Dex1Driver:
            def read_closure(self) -> npt.NDArray[np.float64]:
                if any(item is None for item in latest):
                    raise RuntimeError("Dex1 state topics have not produced samples yet")
                return np.asarray(
                    [
                        packing.dex1_stroke_to_scalar(item.states[0].q, cfg.dex1_stroke)
                        for item in latest
                    ]
                )

            def publish_closure(self, left: float, right: float) -> None:
                for value, command, publisher in zip(
                    (left, right), commands, publishers, strict=True
                ):
                    command.cmds[0].q = packing.dex1_scalar_to_stroke(value, cfg.dex1_stroke)
                    publisher.Write(command)

            def disconnect(self) -> None:
                return None

        return _Dex1Driver()

    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

    topics = ("rt/dex3/left", "rt/dex3/right")
    publishers = [ChannelPublisher(f"{topic}/cmd", HandCmd_) for topic in topics]
    subscribers = [ChannelSubscriber(f"{topic}/state", HandState_) for topic in topics]
    commands = [unitree_hg_msg_dds__HandCmd_(), unitree_hg_msg_dds__HandCmd_()]
    latest = [None, None]
    for index, (publisher, subscriber, command) in enumerate(
        zip(publishers, subscribers, commands, strict=True)
    ):
        publisher.Init()
        for motor_id, motor in enumerate(command.motor_cmd):
            # xr_teleoperate@7dc9aa1a6edb, robot_hand_unitree.py sets RIS
            # status 0x01 with the four-bit motor id for every Dex3 command.
            motor.mode = (motor_id & 0x0F) | (0x01 << 4)
            motor.kp = cast(float, cfg.hand_kp)
            motor.kd = cast(float, cfg.hand_kd)
        subscriber.Init(lambda msg, i=index: latest.__setitem__(i, msg), 10)
    left_closed, right_closed = _dex3_closed_poses(cfg)

    class _Dex3Driver:
        def read_closure(self) -> npt.NDArray[np.float64]:
            if any(item is None for item in latest):
                raise RuntimeError("Dex3 state topics have not produced samples yet")
            return np.asarray(
                (
                    packing.dex3_joints_to_scalar(
                        [motor.q for motor in latest[0].motor_state],
                        packing.DEX3_OPEN_POSE,
                        left_closed,
                    ),
                    packing.dex3_joints_to_scalar(
                        [motor.q for motor in latest[1].motor_state],
                        packing.DEX3_OPEN_POSE,
                        right_closed,
                    ),
                )
            )

        def publish_closure(self, left: float, right: float) -> None:
            poses = (
                packing.dex3_scalar_to_joints(left, packing.DEX3_OPEN_POSE, left_closed),
                packing.dex3_scalar_to_joints(right, packing.DEX3_OPEN_POSE, right_closed),
            )
            for pose, command, publisher in zip(poses, commands, publishers, strict=True):
                for index, value in enumerate(pose):
                    command.motor_cmd[index].q = float(value)
                publisher.Write(command)

        def disconnect(self) -> None:
            return None

    return _Dex3Driver()


def _dex3_closed_poses(cfg: G1Config) -> tuple[Vec, Vec]:
    left = np.asarray(packing.DEX3_LEFT_CLOSED_POSE, dtype=np.float64)
    right = np.asarray(packing.DEX3_RIGHT_CLOSED_POSE, dtype=np.float64)
    left[0] = cfg.dex3_thumb_swing
    right[0] = cfg.dex3_thumb_swing
    return left, right


def decode_jpeg_frame(recv: Callable[[], bytes], cv2_module: Any) -> npt.NDArray[np.uint8]:
    """Receive and decode one raw JPEG payload, returning RGB uint8 pixels."""
    try:
        payload = recv()
    except Exception as exc:
        raise RuntimeError("head camera timed out while waiting for a JPEG frame") from exc
    if not payload:
        raise RuntimeError("head camera returned an empty or stale JPEG frame")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2_module.imdecode(encoded, cv2_module.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("head camera returned an invalid JPEG frame")
    rgb = cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)
    return np.asarray(rgb, dtype=np.uint8)


def _zmq_camera_reader(cfg: G1Config) -> CameraReader:  # pragma: no cover - live transport
    """Build the builtin conflated ZMQ SUB JPEG reader."""
    socket: Any = None

    def reader() -> npt.NDArray[np.uint8]:  # pragma: no cover - live camera transport
        nonlocal socket
        try:
            import cv2
            import zmq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The builtin head camera needs pyzmq and OpenCV. Install with: "
                "pip install pyzmq opencv-python-headless",
                name=exc.name,
            ) from exc
        if socket is None:
            socket = zmq.Context.instance().socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.RCVTIMEO, round(cfg.cam_timeout_s * 1000))
            socket.connect(cfg.cam_server_address)
        return decode_jpeg_frame(socket.recv, cv2)

    return reader


class G1Embodiment:
    """Inspect Robots embodiment for G1 29-DOF arms while balancing."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "unitree_sdk2py": UNITREE_INSTALL_COMMAND,
        "zmq": "pip install pyzmq",
        "cv2": "pip install opencv-python-headless",
    }
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = ()

    def __init__(
        self,
        config: G1Config | None = None,
        *,
        arm_driver_factory: ArmDriverFactory | None = None,
        hand_driver_factory: HandDriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        atexit_module: Any = atexit,
        signal_module: Any = signal,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else G1Config.from_kwargs(**flat)
        self._arm_factory = arm_driver_factory or _default_arm_driver_factory
        self._hand_factory = hand_driver_factory or _default_hand_driver_factory
        self._camera_reader = camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end = poll_end or default_poll_end
        self._clock = clock or time.perf_counter
        self._sleep = sleep_fn or time.sleep
        self._atexit = atexit_module
        self._signal = signal_module
        self._arm: ArmDriver | None = None
        self._hand: HandDriver | None = None
        self._last_arm: Vec | None = None
        self._last_hand: Vec | None = None
        self._weight_enabled = False
        self._backstops_registered = False
        self._previous_sigterm: Any = None
        self._instruction: str | None = None
        self._bound_max_steps: int | None = None
        self._last_publish_time = 0.0
        self.num_steps = 0
        docs = _DOCS
        if self._cfg.docs_extra.strip():
            docs += "\n\n" + self._cfg.docs_extra.strip()
        self.info = EmbodimentInfo(
            name="g1_arms",
            action_space=action_box(self._cfg),
            observation_space=observation_space(),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
        )

    def bind_task(self, envelope: TaskEnvelopeLike) -> None:
        """Store the rollout horizon for operator-facing integrations."""
        self._bound_max_steps = int(envelope.max_steps)

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, seed measured targets, blend in, home, and wait."""
        del seed
        if self._camera_reader is not None and not callable(self._camera_reader):
            raise ValueError("camera_reader must be callable when injected")
        if self._camera_reader is None:
            self._camera_reader = _zmq_camera_reader(self._cfg)
        first = self._arm is None
        if first:
            self._connect()
        arm, hand = self._require_drivers()
        if not self._cfg.unattended:
            self._operator.wait_ready(
                "Arms will blend in and move to the home pose. Stand clear, then press Enter..."
            )
        # Observe after the human wait: the arm may sag or be repositioned
        # while the operator holds the prompt, and the weight blend must hold
        # the pose the arm actually has when motion begins.
        observed_arm = self._read_arm(arm)
        observed_hand = self._read_hands(hand)
        self._last_arm = observed_arm.copy()
        self._last_hand = observed_hand.copy()
        self._last_publish_time = self._clock()
        if not self._weight_enabled:
            self._ramp_weight(0.0, 1.0, observed_arm)
            self._weight_enabled = True
        self._ramp_arms_to(packing.arm_slots(self._cfg.home_pose), weight=1.0)
        self._move_hands_to(
            np.asarray(self._cfg.home_pose, dtype=np.float64)[list(packing.GRIPPER_IDXS)]
        )
        if not self._cfg.unattended:
            self._operator.wait_ready()
        self._instruction = scene.instruction
        self.num_steps = 0
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp, stream speed-limited arm targets, gate hands, pace, and observe."""
        self._require_drivers()
        command = packing.validate_dim(action.data)
        if not np.isfinite(command).all():
            raise ValueError("action must contain only finite values")
        clamped = np.clip(command, self._cfg.low, self._cfg.high)
        self._stream_arm(
            packing.arm_slots(clamped),
            weight=1.0,
            hand_target=np.asarray(clamped[list(packing.GRIPPER_IDXS)]),
        )
        self.num_steps += 1
        try:
            observation = self._observe(self._instruction)
        except BaseException:
            self.close()
            raise
        if not self._cfg.unattended and self._poll_end():
            success = self._operator.confirm_success()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=observation, terminated=False)

    def close(self) -> None:
        """Open hands, park arms, blend weight to zero, and always disconnect."""
        self._bound_max_steps = None
        arm, hand = self._arm, self._hand
        if arm is None and hand is None:
            return
        error: BaseException | None = None
        clean = False
        try:
            if arm is not None and hand is not None:
                observed: Vec | None = None
                try:
                    observed = self._read_arm(arm)
                except BaseException as exc:
                    error = exc
                try:
                    self._move_hands_to(np.ones(2, dtype=np.float64))
                except BaseException as exc:
                    if error is None:
                        error = exc
                target = (
                    packing.arm_slots(self._cfg.rest_pose)
                    if self._cfg.rest_pose is not None
                    else observed
                )
                if target is not None:
                    try:
                        self._ramp_arms_to(target, weight=1.0)
                    except BaseException as exc:
                        if error is None:
                            error = exc
                hold = self._last_arm if self._last_arm is not None else observed
                if hold is not None:
                    try:
                        self._ramp_weight(1.0, 0.0, hold.copy())
                        self._weight_enabled = False
                    except BaseException as exc:
                        if error is None:
                            error = exc
        finally:
            for driver in (hand, arm):
                if driver is not None:
                    try:
                        driver.disconnect()
                    except BaseException as exc:
                        if error is None:
                            error = exc
            self._arm = None
            self._hand = None
            self._last_arm = None
            self._last_hand = None
            clean = error is None
            if clean:
                self._unregister_backstops()
        if error is not None:
            raise error

    def __del__(self) -> None:
        """Best-effort final safety release when normal teardown was skipped."""
        with contextlib.suppress(Exception):
            self.close()

    def _connect(self) -> None:
        arm = self._arm_factory(self._cfg)
        try:
            hand = self._hand_factory(self._cfg)
        except BaseException:
            arm.disconnect()
            raise
        self._arm, self._hand = arm, hand
        try:
            self._register_backstops()
        except BaseException:
            try:
                hand.disconnect()
            finally:
                arm.disconnect()
                self._arm = None
                self._hand = None
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

    def _require_drivers(self) -> tuple[ArmDriver, HandDriver]:
        if self._arm is None or self._hand is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._arm, self._hand

    @staticmethod
    def _read_arm(driver: ArmDriver) -> Vec:
        values: Vec = np.asarray(driver.read_arm_joints(), dtype=np.float64)
        if values.ndim != 1 or values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError(f"arm driver returned invalid joint shape or values: {values.shape}")
        return values

    @staticmethod
    def _read_hands(driver: HandDriver) -> Vec:
        values: Vec = np.asarray(driver.read_closure(), dtype=np.float64)
        if values.ndim != 1 or values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"hand driver returned invalid closure shape or values: {values.shape}"
            )
        return np.clip(values, 0.0, 1.0)

    def _gains(self) -> tuple[Vec, Vec]:
        one_kp = np.asarray([self._cfg.kp_arm] * 4 + [self._cfg.kp_wrist] * 3)
        one_kd = np.asarray([self._cfg.kd_arm] * 4 + [self._cfg.kd_wrist] * 3)
        return np.tile(one_kp, 2), np.tile(one_kd, 2)

    def _publish_arm(self, target: Vec, weight: float) -> None:
        arm, _ = self._require_drivers()
        kp, kd = self._gains()
        arm.publish_arm(target.copy(), weight, kp, kd)
        self._last_arm = target.copy()
        self._pace_publish()

    def _pace_publish(self) -> None:
        elapsed = self._clock() - self._last_publish_time
        self._sleep(max(0.0, 1.0 / self._cfg.stream_hz - elapsed))
        self._last_publish_time = self._clock()

    def _ramp_weight(self, start: float, stop: float, hold: Vec) -> None:
        count = max(1, math.ceil(self._cfg.weight_ramp_s * self._cfg.stream_hz))
        for index in range(count + 1):
            alpha = index / count
            self._publish_arm(hold, (1.0 - alpha) * start + alpha * stop)

    def _stream_arm(self, target: Vec, *, weight: float, hand_target: Vec | None = None) -> None:
        start = self._last_arm
        if start is None:
            raise RuntimeError("published-command baseline is unavailable")
        count = math.ceil(self._cfg.stream_hz / self._cfg.control_hz)
        max_delta = self._cfg.max_joint_speed / self._cfg.stream_hz
        anchor = start.copy()
        sent = start.copy()
        for index in range(1, count + 1):
            desired = anchor + (target - anchor) * (index / count)
            sent = sent + np.clip(desired - sent, -max_delta, max_delta)
            self._publish_arm(sent, weight)
            if hand_target is not None:
                self._publish_hands_if_changed(hand_target)

    def _ramp_arms_to(self, target: Vec, *, weight: float) -> None:
        start = self._last_arm
        if start is None:
            raise RuntimeError("published-command baseline is unavailable")
        max_delta = self._cfg.max_joint_speed / self._cfg.stream_hz
        micro = max(1, math.ceil(float(np.max(np.abs(target - start))) / max_delta))
        count_per_step = math.ceil(self._cfg.stream_hz / self._cfg.control_hz)
        count = max(count_per_step, math.ceil(micro / count_per_step) * count_per_step)
        anchor = start.copy()
        sent = start.copy()
        for index in range(1, count + 1):
            desired = anchor + (target - anchor) * (index / count)
            sent = sent + np.clip(desired - sent, -max_delta, max_delta)
            self._publish_arm(sent, weight)

    def _publish_hands_if_changed(self, target: Vec) -> None:
        _, hand = self._require_drivers()
        previous = self._last_hand
        if previous is None:
            raise RuntimeError("published hand baseline is unavailable")
        if not np.any(np.abs(target - previous) > self._cfg.hand_deadband):
            return
        sent = target.copy()
        if self._cfg.hand_type == "dex1":
            per_publish = self._cfg.dex1_max_speed / self._cfg.dex1_stroke / self._cfg.stream_hz
            sent = previous + np.clip(target - previous, -per_publish, per_publish)
        hand.publish_closure(float(sent[0]), float(sent[1]))
        self._last_hand = sent

    def _move_hands_to(self, target: Vec) -> None:
        """Open or home hands first while respecting Dex1's stream-rate speed cap."""
        previous = self._last_hand
        if previous is None:
            raise RuntimeError("published hand baseline is unavailable")
        if self._cfg.hand_type == "dex3":
            _, hand = self._require_drivers()
            hand.publish_closure(float(target[0]), float(target[1]))
            self._last_hand = target.copy()
            return
        while np.any(np.abs(target - cast(Vec, self._last_hand)) > self._cfg.hand_deadband):
            self._publish_hands_if_changed(target)
            self._pace_publish()

    def _observe(self, instruction: str | None) -> Observation:
        arm, hand = self._require_drivers()
        arm_values = self._read_arm(arm)
        hand_values = self._read_hands(hand)
        packed = packing.pack(
            np.concatenate((arm_values[:7], hand_values[:1])),
            np.concatenate((arm_values[7:], hand_values[1:])),
        )
        body = arm.read_body_state()
        state: dict[str, npt.NDArray[np.float64]] = {packing.STATE_KEY: packed}
        for key, shape in (("left_leg", (6,)), ("right_leg", (6,)), ("waist", (3,))):
            if key not in body:
                raise ValueError(f"arm driver body state is missing {key!r}")
            values = np.asarray(body[key], dtype=np.float64)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"arm driver returned invalid {key} state: {values.shape}")
            state[key] = values.copy()
        reader = self._camera_reader
        if reader is None:  # pragma: no cover - reset always installs one
            raise RuntimeError("camera reader unavailable before reset")
        frame = np.asarray(reader(), dtype=np.uint8)
        observed_at = self._clock()
        return Observation(
            images={"head_cam": frame},
            state=state,
            instruction=instruction,
            image_times={"head_cam": observed_at},
            state_time=observed_at,
        )
