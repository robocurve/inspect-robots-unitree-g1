"""Isaac Lab backend: a simulated G1 walking under a trained velocity policy.

Runs the configured Isaac Lab velocity-tracking task with one environment. The task's
own command generator is frozen, and its base-velocity command is overwritten
with whatever the embodiment asks for. A trained locomotion policy (an
exported TorchScript ``policy.pt`` from rsl_rl) turns that command into joint
targets, so the body walks the way the real controller would.

Pacing: the simulator keeps its own clock. The driver exposes ``sim_time``
and ``advance``, and the embodiment uses them in place of the wall clock and
``time.sleep``, so each control step runs exactly one control period of
physics and the world pauses while the policy (for example an LLM) thinks.
The streamer thread only latches commands; physics advances on the caller's
thread inside ``advance``, because Isaac Sim must stay on one thread.

This module needs Isaac Lab and an NVIDIA RTX GPU; it is exercised on a GPU
host, not in the unit tests.
"""

from __future__ import annotations  # pragma: no cover

import threading  # pragma: no cover
from typing import Any  # pragma: no cover

import numpy as np  # pragma: no cover
import numpy.typing as npt  # pragma: no cover

from inspect_robots_unitree_g1.walking.config import LeggedConfig  # pragma: no cover
from inspect_robots_unitree_g1.walking.driver import LeggedState  # pragma: no cover

_APP: Any = None  # pragma: no cover


def _launch_app(headless: bool) -> Any:  # pragma: no cover - needs Isaac Sim
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=headless, enable_cameras=True).app
    return _APP


#: Coloured boxes placed around the start pose, robot frame (x forward, y left):
#: name, (x, y) in metres, RGB. All sit inside the default 3 m geofence.
PROPS: tuple[
    tuple[str, tuple[float, float], tuple[float, float, float]], ...
] = (  # pragma: no cover
    ("RedBox", (2.0, 1.0), (0.85, 0.1, 0.1)),
    ("BlueBox", (0.8, -1.8), (0.1, 0.25, 0.85)),
    ("GreenBox", (-1.8, 0.4), (0.1, 0.7, 0.2)),
)


def _add_props(env_cfg: Any, sim_utils: Any) -> None:  # pragma: no cover - needs Isaac Sim
    """Add static, collidable coloured boxes (0.4 x 0.4 x 0.8 m) to the scene config."""
    from isaaclab.assets import AssetBaseCfg

    for name, (x, y), rgb in PROPS:
        setattr(
            env_cfg.scene,
            name.lower(),
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(0.4, 0.4, 0.8),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.4)),
            ),
        )


class IsaacLabVelocityDriver:  # pragma: no cover - needs Isaac Sim and a GPU
    """LeggedDriver backed by one Isaac Lab velocity-task environment."""

    def __init__(self, cfg: LeggedConfig) -> None:
        if not cfg.isaac_checkpoint:
            raise ValueError(
                "the isaaclab backend needs isaac_checkpoint: a TorchScript policy.pt "
                "exported by rsl_rl for this robot's velocity task (-E isaac_checkpoint=PATH)"
            )
        _launch_app(cfg.headless)
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401 - registers the gym ids
        import torch
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_tasks.utils import parse_env_cfg

        self._torch = torch
        self._cfg = cfg
        env_cfg = parse_env_cfg(cfg.isaac_task, device="cuda:0", num_envs=1)
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.heading_command = False
        command_cfg.rel_standing_envs = 0.0
        # The command's debug arrows float over the robot and fill the head camera.
        command_cfg.debug_vis = False
        env_cfg.episode_length_s = 1.0e6
        # Start at the env origin facing +x, at rest, so props sit where PROPS says
        # relative to the robot. The task randomises the reset pose by default.
        reset_base = getattr(env_cfg.events, "reset_base", None)
        if reset_base is not None:
            reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {},
            }
        env_cfg.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + cfg.ISAAC_CAMERA_LINK + "/front_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=cfg.ISAAC_CAMERA_POS, rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 50.0)),
            width=cfg.camera_width,
            height=cfg.camera_height,
        )
        if cfg.isaac_props:
            _add_props(env_cfg, sim_utils)
        self._env = gym.make(cfg.isaac_task, cfg=env_cfg)
        self._policy = torch.jit.load(cfg.isaac_checkpoint, map_location="cuda:0").eval()
        obs, _ = self._env.reset()
        self._obs = obs
        self._dt = float(self._env.unwrapped.step_dt)
        self._lock = threading.Lock()
        self._command = (0.0, 0.0, 0.0)
        self.resets = 0
        self._sim_time = 0.0

    def _velocity_term(self) -> Any:
        return self._env.unwrapped.command_manager.get_term("base_velocity")

    def sim_time(self) -> float:
        """Seconds of simulated time since the environment was created."""
        return self._sim_time

    def advance(self, seconds: float) -> None:
        """Run physics for ``seconds`` of sim time under the latest command."""
        torch = self._torch
        target = self._sim_time + float(seconds)
        term = self._velocity_term()
        while self._sim_time + 0.5 * self._dt < target:
            with self._lock:
                command = self._command
            term.vel_command_b[:] = torch.tensor(command, device=term.vel_command_b.device)
            # The observation still holds last tick's command: one physics tick of lag.
            with torch.inference_mode():
                action = self._policy(self._obs["policy"])
            self._obs, _, terminated, truncated, _ = self._env.step(action)
            self._sim_time += self._dt
            if bool(terminated.any()) or bool(truncated.any()):
                self.resets += 1

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Latch the commanded body velocity; physics steps on the reading thread."""
        with self._lock:
            self._command = (float(forward), float(left), float(turn))

    def stand(self) -> None:
        """Latch a zero velocity command."""
        self.send_velocity(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """Read root pose, body velocity and attitude at the current sim time."""
        robot = self._env.unwrapped.scene["robot"].data
        pos = robot.root_pos_w[0].cpu().numpy()
        quat = robot.root_quat_w[0].cpu().numpy()  # w, x, y, z
        w, x, y, z = (float(v) for v in quat)
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        lin = robot.root_lin_vel_b[0].cpu().numpy()
        ang = robot.root_ang_vel_b[0].cpu().numpy()
        return LeggedState(
            pose=np.asarray((pos[0], pos[1], yaw), dtype=np.float64),
            velocity=np.asarray((lin[0], lin[1], ang[2]), dtype=np.float64),
            rpy=np.asarray((roll, pitch, yaw), dtype=np.float64),
        )

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """Return the camera's RGB frame at the current sim time."""
        rgb = self._env.unwrapped.scene["front_cam"].data.output["rgb"][0]
        return np.ascontiguousarray(rgb.cpu().numpy()[..., :3], dtype=np.uint8)

    def disconnect(self) -> None:
        """Close the environment. The SimulationApp stays up for the process."""
        self._env.close()


#: The 14 arm joints in the order the arm IK and the real rt/arm_sdk slots use.
_ARM_JOINTS: tuple[str, ...] = tuple(  # pragma: no cover
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
#: Dex3 finger joints per hand, in Unitree's Dex3 motor order, and their closed
#: poses (open is all zeros), from this package's packing.py (g1_arms).
_DEX3_JOINTS: tuple[str, ...] = (  # pragma: no cover
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
_DEX3_CLOSED = {  # pragma: no cover
    "left": (0.0, 0.6, 0.8, -1.2, -1.4, -1.2, -1.4),
    "right": (0.0, -0.6, -0.8, 1.2, 1.4, 1.2, 1.4),
}
#: Dex1-1 sliding finger joints per hand (unitree_ros dex1_1 model), and
#: their open and closed positions in metres.
_DEX1_JOINTS: tuple[str, ...] = ("dex1_finger_joint_1", "dex1_finger_joint_2")  # pragma: no cover
_DEX1_OPEN, _DEX1_CLOSED = 0.0245, -0.02  # pragma: no cover
_HIP_HEIGHT = 0.72  # pragma: no cover - Isaac Lab's G1 lower-body standing default


class IsaacLabG1MobileDriver:  # pragma: no cover - needs Isaac Sim and a GPU
    """Walking plus both arms on Isaac Lab's 29-DoF G1 locomanipulation scene.

    Legs: Isaac Lab's pretrained Agile lower-body policy, commanded by body
    velocity (the stand-in for Unitree's locomotion controller). Arms: this
    package's numpy IK on the real 29-DoF arm model, sent as joint position
    targets (the stand-in for rt/arm_sdk). Hands: Dex3 finger targets
    interpolated between open and closed by grip.
    """

    def __init__(self, cfg: Any) -> None:
        from inspect_robots_unitree_g1.walking.arm_kinematics import G1_ARMS, with_tool_offset

        _launch_app(cfg.headless)
        import gymnasium as gym
        import isaaclab.envs.mdp as base_mdp
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401 - registers the gym ids
        import torch
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (  # noqa: E501
            LocomanipulationG1EnvCfg,
        )

        self._torch = torch
        self._cfg = cfg
        self._arms = with_tool_offset(G1_ARMS, cfg.tool_offset_m)
        env_cfg = LocomanipulationG1EnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = "cuda:0"
        env_cfg.episode_length_s = 1.0e6
        dex1 = cfg.hands == "dex1"
        if dex1:
            # The walking policy reads body joints only, so the hands can change.
            from isaaclab.actuators import ImplicitActuatorCfg

            if not cfg.isaac_robot_usd:
                raise ValueError("hands=dex1 in Isaac needs -E isaac_robot_usd=PATH")
            robot_cfg = env_cfg.scene.robot
            actuators = {k: v for k, v in robot_cfg.actuators.items() if k != "hands"}
            actuators["dex1"] = ImplicitActuatorCfg(
                joint_names_expr=[".*_dex1_finger_joint_.*"],
                effort_limit=50.0,
                velocity_limit=0.5,
                stiffness=2000.0,
                damping=50.0,
            )
            env_cfg.scene.robot = robot_cfg.replace(
                spawn=robot_cfg.spawn.replace(usd_path=cfg.isaac_robot_usd), actuators=actuators
            )
            # The task's own hand-state observation looks for Dex3 joint names.
            env_cfg.observations.policy.hand_joint_state.params = {
                "joint_names": [".*_dex1_finger_joint_.*"]
            }
            hand_joints = [f"{s}_{n}" for s in ("left", "right") for n in _DEX1_JOINTS]
            per_hand = len(_DEX1_JOINTS)
            open_q = np.full(2 * per_hand, _DEX1_OPEN)
            closed_q = np.full(2 * per_hand, _DEX1_CLOSED)
        else:
            hand_joints = [f"{s}_hand_{n}_joint" for s in ("left", "right") for n in _DEX3_JOINTS]
            per_hand = len(_DEX3_JOINTS)
            open_q = np.zeros(2 * per_hand)
            closed_q = np.asarray((*_DEX3_CLOSED["left"], *_DEX3_CLOSED["right"]))
        self._per_hand = per_hand
        # Arms and hands as plain joint targets, in our order.
        env_cfg.actions.upper_body_ik = base_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[*_ARM_JOINTS, *hand_joints],
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        if cfg.isaac_props:
            # Walk-then-reach layout: the robot starts about a metre back from the
            # packing table (front edge at y = 0.149), facing it, and a small red
            # cube sits 15 cm in from the edge, in front of the right hand, in
            # place of the task's steering wheel.
            from isaaclab.assets import RigidObjectCfg

            env_cfg.scene.robot.init_state.pos = (0.0, -0.9, 0.75)
            env_cfg.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, 0.30, 0.73)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.06, 0.06, 0.06),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
                ),
            )
        # Only a fall ends anything; the pick-and-place success and drop checks go.
        for name in ("object_dropping", "success", "time_out"):
            if hasattr(env_cfg.terminations, name):
                setattr(env_cfg.terminations, name, None)
        env_cfg.scene.head_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + cfg.ISAAC_CAMERA_LINK + "/head_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=cfg.ISAAC_CAMERA_POS, rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 50.0)),
            width=cfg.camera_width,
            height=cfg.camera_height,
        )
        self._env = gym.make("Isaac-PickPlace-Locomanipulation-G1-Abs-v0", cfg=env_cfg).unwrapped
        self._env.reset()
        robot = self._env.scene["robot"]
        self._arm_ids = [robot.joint_names.index(name) for name in _ARM_JOINTS]
        self._dt = float(self._env.step_dt)
        self._lock = threading.Lock()
        self._velocity = (0.0, 0.0, 0.0)
        q = robot.data.joint_pos[0, self._arm_ids].cpu().numpy().astype(np.float64)
        self._q_solution = [q[:7].copy(), q[7:].copy()]
        self._grips = np.zeros(2, dtype=np.float64)
        self._open = open_q
        self._closed = closed_q
        self._sim_time = 0.0
        self.resets = 0

    # -- clock --------------------------------------------------------------

    def sim_time(self) -> float:
        """Seconds of simulated time since the environment was created."""
        return self._sim_time

    def advance(self, seconds: float) -> None:
        """Run physics for ``seconds`` of sim time under the latest commands."""
        torch = self._torch
        target = self._sim_time + float(seconds)
        while self._sim_time + 0.5 * self._dt < target:
            with self._lock:
                velocity = self._velocity
                arm_q = np.concatenate(self._q_solution)
                fingers = self._open + (self._closed - self._open) * np.repeat(
                    self._grips, self._per_hand
                )
            action = np.concatenate((arm_q, fingers, velocity, (_HIP_HEIGHT,)))
            tensor = torch.tensor(action, dtype=torch.float32, device=self._env.device)
            _, _, terminated, truncated, _ = self._env.step(tensor.unsqueeze(0))
            self._sim_time += self._dt
            if bool(terminated.any()) or bool(truncated.any()):
                self.resets += 1

    # -- walking ------------------------------------------------------------

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        """Latch the commanded body velocity."""
        with self._lock:
            self._velocity = (float(forward), float(left), float(turn))

    def stand(self) -> None:
        """Latch zero velocity."""
        self.send_velocity(0.0, 0.0, 0.0)

    def read_state(self) -> LeggedState:
        """Root pose, body velocity and attitude at the current sim time."""
        data = self._env.scene["robot"].data
        pos = data.root_pos_w[0].cpu().numpy()
        w, x, y, z = (float(v) for v in data.root_quat_w[0].cpu().numpy())
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        lin = data.root_lin_vel_b[0].cpu().numpy()
        ang = data.root_ang_vel_b[0].cpu().numpy()
        return LeggedState(
            pose=np.asarray((pos[0], pos[1], yaw), dtype=np.float64),
            velocity=np.asarray((lin[0], lin[1], ang[2]), dtype=np.float64),
            rpy=np.asarray((roll, pitch, yaw), dtype=np.float64),
        )

    def read_frame(self) -> npt.NDArray[np.uint8]:
        """The head camera's RGB frame at the current sim time."""
        rgb = self._env.scene["head_cam"].data.output["rgb"][0]
        return np.ascontiguousarray(rgb.cpu().numpy()[..., :3], dtype=np.uint8)

    # -- hands --------------------------------------------------------------

    def send_hands(self, targets: Any, grips: Any) -> Any:
        """Solve IK (warm start) and latch arm and finger targets; return reachable palms."""
        from inspect_robots_unitree_g1.walking.arm_kinematics import fk, ik

        reached = np.zeros((2, 3), dtype=np.float64)
        solutions = []
        for side, arm in enumerate(self._arms):
            with self._lock:
                start = self._q_solution[side].copy()
            solution, _ = ik(arm, np.asarray(targets[side]), start)
            solutions.append(solution)
            reached[side] = fk(arm, solution)
        with self._lock:
            self._q_solution = solutions
            self._grips = np.clip(np.asarray(grips, dtype=np.float64), 0.0, 1.0)
        return reached

    def read_hands(self) -> tuple[Any, Any]:
        """Palm positions from the simulated arm joints, and the grips."""
        from inspect_robots_unitree_g1.walking.arm_kinematics import fk

        q = self._env.scene["robot"].data.joint_pos[0, self._arm_ids].cpu().numpy()
        q = q.astype(np.float64)
        palms = np.stack([fk(self._arms[0], q[:7]), fk(self._arms[1], q[7:])])
        with self._lock:
            return palms, self._grips.copy()

    def release_hands(self) -> bool:
        """Simulation has no controller to hand back to; open the hands."""
        with self._lock:
            self._grips = np.zeros(2, dtype=np.float64)
        return True

    def disconnect(self) -> None:
        """Close the environment. The SimulationApp stays up for the process."""
        self._env.close()
