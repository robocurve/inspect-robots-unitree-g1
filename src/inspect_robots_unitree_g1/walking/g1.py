"""Unitree G1 walking backend over ``unitree_sdk2py``'s locomotion RPC client.

The G1's built-in locomotion controller ("sport" service) takes a body
velocity with a duration through ``LocoClient.SetVelocity``. Every command
carries ``velocity_hold_s``, so the robot stops on its own when commands stop
arriving, including when this process is killed outright.

This backend never changes the robot's FSM mode. The operator stands the
robot and puts it in walking mode with the Unitree remote; connecting refuses
a robot that is in zero-torque or damping mode.

State comes from two DDS topics: ``rt/lowstate`` (IMU roll, pitch, yaw;
required) and the odometry topic (``rt/odommodestate`` by default: position,
velocity and yaw rate). When odometry is absent or stale the pose is
dead-reckoned from the commanded velocity and the IMU yaw.

Not yet verified on a physical G1: the odometry topic name, its velocity
frame and the FSM ids should be checked on the robot first.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots_unitree_g1 import packing
from inspect_robots_unitree_g1._unitree import _load_unitree
from inspect_robots_unitree_g1.walking.arm_kinematics import (
    G1_ARMS,
    ArmChain,
    fk,
    ik,
    with_tool_offset,
)
from inspect_robots_unitree_g1.walking.config import G1Config, G1MobileConfig
from inspect_robots_unitree_g1.walking.driver import LeggedDriver, LeggedState

Vec = npt.NDArray[np.float64]
CameraReader = Callable[[], npt.NDArray[np.uint8]]


#: FSM ids in which the robot is limp, from unitree_sdk2py LocoClient:
#: ZeroTorque() sets 0 and Damp() sets 1.
LIMP_FSM_IDS: frozenset[int] = frozenset({0, 1})


class LocoClientLike(Protocol):
    """The slice of ``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient`` used here."""

    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0) -> int:
        """Hold a body velocity for ``duration`` seconds; returns 0 on success."""
        ...

    def GetFsmId(self) -> tuple[int, Any]:
        """Return ``(code, fsm_id)`` for the controller's current mode."""
        ...


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class G1LocoDriver:
    """LeggedDriver-compatible backend for the G1's locomotion service.

    Message callbacks (``on_lowstate``, ``on_odom``) run on DDS threads; every
    read and write of shared state holds one lock.
    """

    def __init__(
        self,
        cfg: G1Config,
        *,
        loco: LocoClientLike,
        clock: Callable[[], float],
        camera_reader: CameraReader | None = None,
        closers: Sequence[Callable[[], None]] = (),
    ) -> None:
        self._cfg = cfg
        self._loco = loco
        self._clock = clock
        self._camera_reader = camera_reader
        self._closers = tuple(closers)
        self._lock = threading.Lock()
        self._rpy: Vec | None = None
        self._odom: tuple[float, Vec, Vec] | None = None  # (time, pose, body velocity)
        self._command = np.zeros(3, dtype=np.float64)
        self._dead_pose = np.zeros(3, dtype=np.float64)
        self._dead_time = clock()
        self.fsm_id: int | None = None

    # ------------------------------------------------------------------ connect

    def check_ready(self) -> int:
        """Refuse a limp robot. Returns the FSM id the robot reported."""
        code, fsm_id = self._loco.GetFsmId()
        if code != 0:
            raise RuntimeError(f"G1 GetFsmId failed with code {code}; is the sport service up?")
        fsm = int(fsm_id)
        if fsm in LIMP_FSM_IDS:
            raise RuntimeError(
                f"G1 is limp (FSM id {fsm}: zero torque or damping). Stand it up and "
                "start walking mode with the Unitree remote, then retry."
            )
        self.fsm_id = fsm
        return fsm

    # ------------------------------------------------------------------ DDS callbacks

    def on_lowstate(self, msg: Any) -> None:
        """Store the IMU attitude from an ``rt/lowstate`` sample."""
        rpy = np.asarray(msg.imu_state.rpy, dtype=np.float64)
        with self._lock:
            self._rpy = rpy

    def on_odom(self, msg: Any) -> None:
        """Store pose and body velocity from a SportModeState_ odometry sample.

        Position and velocity are taken to be in the odometry (world) frame;
        the velocity is rotated into the body frame with the sample's yaw.
        """
        yaw = float(msg.imu_state.rpy[2])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy = float(msg.velocity[0]), float(msg.velocity[1])
        pose = np.asarray((msg.position[0], msg.position[1], yaw), dtype=np.float64)
        velocity = np.asarray(
            (cos * vx + sin * vy, -sin * vx + cos * vy, float(msg.yaw_speed)), dtype=np.float64
        )
        with self._lock:
            self._odom = (self._clock(), pose, velocity)

    # ------------------------------------------------------------------ LeggedDriver

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Command a body velocity that lapses after ``velocity_hold_s`` on the robot."""
        with self._lock:
            self._advance_locked()
            self._command = np.asarray((forward, left, turn), dtype=np.float64)
        code = self._loco.SetVelocity(
            float(forward), float(left), float(turn), self._cfg.velocity_hold_s
        )
        if code != 0:
            raise RuntimeError(f"G1 SetVelocity failed with code {code}")

    def stand(self) -> None:
        """Command zero velocity. The robot keeps balancing in place."""
        self.send_velocity(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """Odometry when fresh, otherwise dead reckoning on the IMU yaw."""
        with self._lock:
            if self._rpy is None:
                raise RuntimeError("rt/lowstate has not produced a sample yet")
            self._advance_locked()
            rpy = self._rpy.copy()
            odom = self._odom
            if odom is not None and self._clock() - odom[0] <= self._cfg.odom_stale_s:
                return LeggedState(pose=odom[1].copy(), velocity=odom[2].copy(), rpy=rpy)
            return LeggedState(pose=self._dead_pose.copy(), velocity=self._command.copy(), rpy=rpy)

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Return a head camera frame."""
        if self._camera_reader is None:
            raise RuntimeError("no G1 camera reader configured")
        return self._camera_reader()

    def disconnect(self) -> None:
        """Stop walking, then close every subscriber. Always tries every closer."""
        error: BaseException | None = None
        try:
            self.stand()
        except BaseException as exc:
            error = exc
        for close in self._closers:
            try:
                close()
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    # ------------------------------------------------------------------ helpers

    def _advance_locked(self) -> None:
        """Integrate the commanded velocity using the IMU yaw when there is one."""
        now = self._clock()
        dt = max(0.0, now - self._dead_time)
        self._dead_time = now
        yaw = float(self._rpy[2]) if self._rpy is not None else float(self._dead_pose[2])
        cos, sin = math.cos(yaw), math.sin(yaw)
        forward, left, turn = (float(v) for v in self._command)
        self._dead_pose = np.asarray(
            (
                self._dead_pose[0] + dt * (cos * forward - sin * left),
                self._dead_pose[1] + dt * (sin * forward + cos * left),
                yaw if self._rpy is not None else _wrap(self._dead_pose[2] + dt * turn),
            ),
            dtype=np.float64,
        )


#: rt/lowstate and rt/arm_sdk motor slots for the arms (left shoulder pitch
#: through left wrist yaw, then the right arm), and the arm-sdk blend weight
#: slot, per unitree_sdk2_python example/g1/high_level/g1_arm7_sdk_dds_example.py.
ARM_MOTOR_INDICES: tuple[int, ...] = tuple(range(15, 29))
ARM_WEIGHT_INDEX = 29


class ArmPublisher(Protocol):
    """Publishes one rt/arm_sdk command: 14 arm targets, the blend weight, and gains."""

    def publish(self, q14: Vec, weight: float, kp: Vec, kd: Vec) -> None:
        """Send arm joint targets with gains and the arm-sdk blend weight."""
        ...


class GripperIO(Protocol):
    """Both Dex1-1 grippers: motor strokes in radians, left then right."""

    def read_strokes(self) -> Vec:
        """Measured strokes ``(2,)``; raises RuntimeError before the first state sample."""
        ...

    def publish_strokes(self, strokes: Vec) -> None:
        """Command stroke targets ``(2,)``."""
        ...


def grip_to_dex1_stroke(grip: npt.ArrayLike, stroke: float) -> Vec:
    """Grip 0 (open) to 1 (closed) to Dex1 stroke, through g1_arms' open-positive mapping."""
    grips = np.asarray(grip, dtype=np.float64).reshape(-1)
    return np.asarray([packing.dex1_scalar_to_stroke(1.0 - g, stroke) for g in grips])


def dex1_stroke_to_grip(value: npt.ArrayLike, stroke: float) -> Vec:
    """Dex1 stroke back to grip 0 (open) to 1 (closed)."""
    strokes = np.asarray(value, dtype=np.float64).reshape(-1)
    return np.asarray([1.0 - packing.dex1_stroke_to_scalar(v, stroke) for v in strokes])


class G1MobileSdkDriver(G1LocoDriver):
    """The G1 walking driver plus both arms over rt/arm_sdk, by palm position.

    The arm-sdk weight blends the arms from the locomotion controller to these
    commands: it rises from 0 to 1 over ``arm_weight_ramp_s`` once arm commands
    start, while the first command holds the measured pose, and falls back to
    0 on release. Every tick solves IK warm-started from the last solution,
    then moves each joint command at most ``arm_joint_step`` toward it.

    With ``grippers`` (Dex1-1), grip targets become stroke commands, starting
    from the measured stroke and moving at most ``dex1_max_speed`` rad/s.
    Release leaves the grippers holding their last command, so a held object
    is not dropped. Without grippers, reported grips stay 0.
    """

    def __init__(
        self,
        cfg: G1MobileConfig,
        *,
        loco: LocoClientLike,
        clock: Callable[[], float],
        arm_publisher: ArmPublisher,
        camera_reader: CameraReader | None = None,
        closers: Sequence[Callable[[], None]] = (),
        arms: tuple[ArmChain, ArmChain] | None = None,
        grippers: GripperIO | None = None,
    ) -> None:
        super().__init__(cfg, loco=loco, clock=clock, camera_reader=camera_reader, closers=closers)
        self._mobile_cfg = cfg
        self._arm_publisher = arm_publisher
        self._arms = arms if arms is not None else with_tool_offset(G1_ARMS, cfg.tool_offset_m)
        self._grippers = grippers
        self._stroke_cmd: Vec | None = None
        self._arm_q: Vec | None = None
        self._q_cmd: Vec | None = None
        self._q_solution: list[Vec] | None = None
        self._weight = 0.0
        one_kp = (cfg.kp_arm,) * 4 + (cfg.kp_wrist,) * 3
        one_kd = (cfg.kd_arm,) * 4 + (cfg.kd_wrist,) * 3
        self._kp = np.asarray(one_kp * 2, dtype=np.float64)
        self._kd = np.asarray(one_kd * 2, dtype=np.float64)

    @property
    def weight(self) -> float:
        """Current arm-sdk blend weight, 0 (robot's controller) to 1 (these commands)."""
        return self._weight

    def on_lowstate(self, msg: Any) -> None:
        """Store the IMU attitude and the 14 measured arm joint angles."""
        q = np.asarray([msg.motor_state[i].q for i in ARM_MOTOR_INDICES], dtype=np.float64)
        super().on_lowstate(msg)
        with self._lock:
            self._arm_q = q

    def _measured_arm_q(self) -> Vec:
        with self._lock:
            if self._arm_q is None:
                raise RuntimeError("rt/lowstate has not produced arm joint angles yet")
            return self._arm_q.copy()

    def read_hands(self) -> tuple[Vec, Vec]:
        """Palm positions from the measured arm joints, and measured grips (0 without grippers)."""
        q = self._measured_arm_q()
        palms = np.stack(
            [fk(arm, q[7 * side : 7 * side + 7]) for side, arm in enumerate(self._arms)]
        )
        if self._grippers is None:
            return palms, np.zeros(2, dtype=np.float64)
        strokes = self._measured_strokes()
        return palms, dex1_stroke_to_grip(strokes, self._mobile_cfg.dex1_stroke)

    def _measured_strokes(self) -> Vec:
        assert self._grippers is not None
        strokes = np.asarray(self._grippers.read_strokes(), dtype=np.float64)
        if strokes.shape != (2,) or not np.all(np.isfinite(strokes)):
            raise ValueError(f"Dex1 grippers returned invalid strokes: {strokes!r}")
        return strokes

    def _send_grips(self, grips: Vec) -> None:
        if self._grippers is None:
            return
        cfg = self._mobile_cfg
        if self._stroke_cmd is None:
            self._stroke_cmd = self._measured_strokes()
        desired = grip_to_dex1_stroke(grips, cfg.dex1_stroke)
        step = cfg.dex1_max_speed / cfg.stream_hz
        self._stroke_cmd = self._stroke_cmd + np.clip(desired - self._stroke_cmd, -step, step)
        self._grippers.publish_strokes(self._stroke_cmd.copy())

    def send_hands(self, targets: Vec, grips: Vec) -> Vec:
        """Solve, rate-limit, ramp the weight toward 1, publish; return reachable palms."""
        self._send_grips(np.asarray(grips, dtype=np.float64))
        if self._q_cmd is None:
            self._q_cmd = self._measured_arm_q()
            self._q_solution = [self._q_cmd[:7].copy(), self._q_cmd[7:].copy()]
        assert self._q_solution is not None
        reached = np.zeros((2, 3), dtype=np.float64)
        for side, arm in enumerate(self._arms):
            solution, _ = ik(arm, np.asarray(targets[side]), self._q_solution[side])
            self._q_solution[side] = solution
            reached[side] = fk(arm, solution)
        desired = np.concatenate(self._q_solution)
        step = self._mobile_cfg.arm_joint_step
        self._q_cmd = self._q_cmd + np.clip(desired - self._q_cmd, -step, step)
        ramp = 1.0 / (self._mobile_cfg.arm_weight_ramp_s * self._mobile_cfg.stream_hz)
        self._weight = min(1.0, self._weight + ramp)
        if self._weight > 1.0 - 1e-9:
            self._weight = 1.0
        self._arm_publisher.publish(self._q_cmd.copy(), self._weight, self._kp, self._kd)
        return reached

    def release_hands(self) -> bool:
        """Hold the last command and ramp the weight down one tick; True at zero."""
        if self._q_cmd is None:
            self._weight = 0.0
            self._stroke_cmd = None
            return True
        ramp = 1.0 / (self._mobile_cfg.arm_weight_ramp_s * self._mobile_cfg.stream_hz)
        self._weight = max(0.0, self._weight - ramp)
        if self._weight < 1e-9:  # absorb float residue from repeated subtraction
            self._weight = 0.0
        self._arm_publisher.publish(self._q_cmd.copy(), self._weight, self._kp, self._kd)
        if self._weight > 0.0:
            return False
        self._q_cmd = None
        self._q_solution = None
        self._stroke_cmd = None
        return True


_DDS_INITIALIZED = False


def build_g1_sdk_driver(cfg: G1Config) -> LeggedDriver:  # pragma: no cover - hardware
    """Open DDS, the locomotion client, state subscribers and the head camera."""
    global _DDS_INITIALIZED
    import time

    _load_unitree()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    if not _DDS_INITIALIZED:
        ChannelFactoryInitialize(0, cfg.network_interface)
        _DDS_INITIALIZED = True
    loco = LocoClient()
    loco.SetTimeout(cfg.rpc_timeout_s)
    loco.Init()
    lowstate = ChannelSubscriber("rt/lowstate", LowState_)
    odom = ChannelSubscriber(cfg.odom_topic, SportModeState_)
    driver = G1LocoDriver(
        cfg,
        loco=loco,
        clock=time.monotonic,
        camera_reader=_zmq_jpeg_reader(cfg),
        closers=(lowstate.Close, odom.Close),
    )
    lowstate.Init(driver.on_lowstate, 10)
    odom.Init(driver.on_odom, 10)
    try:
        driver.check_ready()
        deadline = time.monotonic() + 2.0
        while True:
            try:
                driver.read_state()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
    except BaseException:
        lowstate.Close()
        odom.Close()
        raise
    return driver


def build_g1_mobile_sdk_driver(cfg: G1MobileConfig) -> LeggedDriver:  # pragma: no cover - hardware
    """Open everything ``build_g1_sdk_driver`` opens, plus the rt/arm_sdk publisher."""
    global _DDS_INITIALIZED
    import time

    _load_unitree()
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    if not _DDS_INITIALIZED:
        ChannelFactoryInitialize(0, cfg.network_interface)
        _DDS_INITIALIZED = True
    loco = LocoClient()
    loco.SetTimeout(cfg.rpc_timeout_s)
    loco.Init()
    arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    arm_pub.Init()
    command = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()

    class _ArmSdk:
        def publish(self, q14: Vec, weight: float, kp: Vec, kd: Vec) -> None:
            for slot, index in enumerate(ARM_MOTOR_INDICES):
                motor = command.motor_cmd[index]
                motor.q = float(q14[slot])
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = float(kp[slot])
                motor.kd = float(kd[slot])
            command.motor_cmd[ARM_WEIGHT_INDEX].q = float(weight)
            command.crc = crc.Crc(command)
            arm_pub.Write(command)

    lowstate = ChannelSubscriber("rt/lowstate", LowState_)
    odom = ChannelSubscriber(cfg.odom_topic, SportModeState_)
    grippers, gripper_closers = _dex1_grippers(cfg) if cfg.hands == "dex1" else (None, ())
    driver = G1MobileSdkDriver(
        cfg,
        loco=loco,
        clock=time.monotonic,
        arm_publisher=_ArmSdk(),
        camera_reader=_zmq_jpeg_reader(cfg),
        closers=(lowstate.Close, odom.Close, *gripper_closers),
        grippers=grippers,
    )
    lowstate.Init(driver.on_lowstate, 10)
    odom.Init(driver.on_odom, 10)
    try:
        driver.check_ready()
        deadline = time.monotonic() + 2.0
        while True:
            try:
                driver.read_state()
                driver.read_hands()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
    except BaseException:
        lowstate.Close()
        odom.Close()
        for close in gripper_closers:
            close()
        raise
    return driver


#: Dex1-1 DDS topic prefixes, left then right; commands on ``/cmd``, state on
#: ``/state`` (unitree_go MotorCmds_ / MotorStates_, one motor each), as used
#: by g1_arms in this package and by xr_teleoperate.
DEX1_TOPICS: tuple[str, str] = ("rt/dex1/left", "rt/dex1/right")


def _dex1_grippers(
    cfg: G1MobileConfig,
) -> tuple[GripperIO, tuple[Callable[[], None], ...]]:  # pragma: no cover - hardware
    """Open both Dex1-1 command publishers and state subscribers."""
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    lock = threading.Lock()
    latest: list[Any] = [None, None]
    publishers = [ChannelPublisher(f"{topic}/cmd", MotorCmds_) for topic in DEX1_TOPICS]
    subscribers = [ChannelSubscriber(f"{topic}/state", MotorStates_) for topic in DEX1_TOPICS]
    commands = [MotorCmds_(), MotorCmds_()]

    def store(index: int, msg: Any) -> None:
        with lock:
            latest[index] = msg

    for index, (publisher, subscriber, command) in enumerate(
        zip(publishers, subscribers, commands, strict=True)
    ):
        publisher.Init()
        command.cmds = [unitree_go_msg_dds__MotorCmd_()]
        command.cmds[0].kp = cfg.dex1_kp
        command.cmds[0].kd = cfg.dex1_kd
        subscriber.Init(lambda msg, i=index: store(i, msg), 10)

    class _Dex1:
        def read_strokes(self) -> Vec:
            with lock:
                if any(item is None for item in latest):
                    raise RuntimeError("Dex1 state topics have not produced samples yet")
                return np.asarray([item.states[0].q for item in latest], dtype=np.float64)

        def publish_strokes(self, strokes: Vec) -> None:
            for value, command, publisher in zip(strokes, commands, publishers, strict=True):
                command.cmds[0].q = float(value)
                publisher.Write(command)

    return _Dex1(), tuple(subscriber.Close for subscriber in subscribers)


def _zmq_jpeg_reader(cfg: G1Config) -> CameraReader:  # pragma: no cover - live camera
    """Read the head camera's JPEG stream, the same transport and decoder as ``g1_arms``."""
    from inspect_robots_unitree_g1.embodiment import decode_jpeg_frame

    socket: Any = None

    def reader() -> npt.NDArray[np.uint8]:
        nonlocal socket
        try:
            import cv2
            import zmq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The G1 head camera needs pyzmq and OpenCV: "
                "pip install pyzmq opencv-python-headless",
                name=exc.name,
            ) from exc
        if socket is None:
            socket = zmq.Context.instance().socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.RCVTIMEO, round(cfg.cam_timeout_s * 1000))
            socket.connect(cfg.cam_server_address)
        frame = decode_jpeg_frame(socket.recv, cv2)
        if frame.shape[:2] != (cfg.camera_height, cfg.camera_width):
            frame = cv2.resize(frame, (cfg.camera_width, cfg.camera_height))
        return np.asarray(frame, dtype=np.uint8)

    return reader
