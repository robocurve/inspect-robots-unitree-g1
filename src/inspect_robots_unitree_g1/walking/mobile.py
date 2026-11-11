"""The G1 walking and using both hands in one trial (embodiment ``g1_mobile``).

Each action is 11 per-step displacements: the three walking dimensions of
``g1_base``, then each palm's (x, y, z) in the torso frame, then each grip.
The embodiment owns the hand targets: it adds each step's displacement,
clamps the result to a workspace box, and streams the targets to the driver
on every tick. The driver turns targets into joint commands (inverse
kinematics on its own arm model) and returns where the solution actually
puts each palm; the embodiment adopts that as the new target, so a target
pushed out of reach never winds up beyond it.
"""

from __future__ import annotations

import math
import threading
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from inspect_robots_unitree_g1.walking.arm_kinematics import (
    G1_ARMS,
    ArmChain,
    fk,
    ik,
    with_tool_offset,
)
from inspect_robots_unitree_g1.walking.config import (
    GRIPS_KEY,
    HANDS_KEY,
    G1MobileConfig,
    LeggedConfig,
)
from inspect_robots_unitree_g1.walking.driver import KinematicLeggedDriver, LeggedDriver
from inspect_robots_unitree_g1.walking.embodiment import G1Embodiment

Vec = npt.NDArray[np.float64]

#: Where the palms go at reset, torso frame: in front of the chest, apart.
HOME_HANDS = np.asarray(((0.25, 0.20, 0.05), (0.25, -0.20, 0.05)), dtype=np.float64)


@runtime_checkable
class HandsDriver(Protocol):
    """What a walking backend must add to drive both arms by palm position."""

    def send_hands(self, targets: Vec, grips: Vec) -> Vec:
        """Command palm targets ``(2, 3)`` and grips ``(2,)``; return the reachable palms."""
        ...

    def read_hands(self) -> tuple[Vec, Vec]:
        """Measured palm positions ``(2, 3)`` and grips ``(2,)``."""
        ...

    def release_hands(self) -> bool:
        """Advance handing the arms back to the robot's controller; True when done."""
        ...

    def engage_hands(self) -> bool:
        """Advance taking the arms over while holding their measured pose; True when done.

        The first call of each takeover reseeds the command baseline from the
        measured joints, so every reset starts from where the arms really are.
        """
        ...


class KinematicMobileDriver(KinematicLeggedDriver):
    """The kinematic stand-in plus two arms that follow IK instantly.

    It runs the real G1 arm IK, so mock runs exercise reachability and the
    target projection exactly as the real driver would.
    """

    def __init__(
        self,
        cfg: LeggedConfig,
        *,
        clock: Any,
        response_time_s: float = 0.15,
        arms: tuple[ArmChain, ArmChain] | None = None,
    ) -> None:
        super().__init__(cfg, clock=clock, response_time_s=response_time_s)
        if arms is None:
            offset = cfg.tool_offset_m if isinstance(cfg, G1MobileConfig) else 0.0
            arms = with_tool_offset(G1_ARMS, offset)
        self._arms = arms
        self._q = [np.asarray(arm.nominal, dtype=np.float64) for arm in arms]
        self._grips = np.zeros(2, dtype=np.float64)
        self._arm_lock = threading.Lock()

    def send_hands(self, targets: Vec, grips: Vec) -> Vec:
        """Solve IK for each palm and snap the arm there."""
        reached = np.zeros((2, 3), dtype=np.float64)
        with self._arm_lock:
            for side, arm in enumerate(self._arms):
                self._q[side], _ = ik(arm, targets[side], self._q[side])
                reached[side] = fk(arm, self._q[side])
            self._grips = np.clip(np.asarray(grips, dtype=np.float64), 0.0, 1.0)
        return reached

    def read_hands(self) -> tuple[Vec, Vec]:
        """Palm positions from the current joint angles, and the grips."""
        with self._arm_lock:
            palms = np.stack([fk(arm, q) for arm, q in zip(self._arms, self._q, strict=True)])
            return palms, self._grips.copy()

    def engage_hands(self) -> bool:
        """The stand-in has no controller to take over from."""
        return True

    def release_hands(self) -> bool:
        """Drop the arms back to their nominal pose and open the grips."""
        with self._arm_lock:
            self._q = [np.asarray(arm.nominal, dtype=np.float64) for arm in self._arms]
            self._grips = np.zeros(2, dtype=np.float64)
        return True


_MOBILE_DOCS = """A Unitree G1 humanoid that can walk and use both arms in one trial.
Its own locomotion controller keeps it balanced; you steer the body and move
each hand. Each action is how far things should move during one control step:
- base_forward: metres toward the front of the torso (negative walks backward).
- base_left: metres toward the robot's left side (negative steps right).
- base_turn: radians of turn, positive counter-clockwise seen from above.
- left_hand_x: metres the left palm moves forward, in the torso frame.
- left_hand_y: metres the left palm moves toward the robot's left.
- left_hand_z: metres the left palm moves up.
- right_hand_x: metres the right palm moves forward, in the torso frame.
- right_hand_y: metres the right palm moves toward the robot's left.
- right_hand_z: metres the right palm moves up.
- left_grip: change in left grip, 0 open to 1 closed.
- right_grip: change in right grip, 0 open to 1 closed.
hand_pos in the state is (left x, y, z, right x, y, z) in metres from the
torso origin (roughly the waist): x forward, y left, z up. Hands start in
front of the chest, about 0.25 m forward, 0.2 m out to each side and 0.05 m
up. Each palm stays inside a box from 0.18 to 0.40 m forward, 0.05 to 0.38 m
out to its own side, and -0.05 to 0.30 m up; a move past the box or out of
reach stops at the edge, and hand_pos shows where the palm really is.
grip in the state is each hand's grip, 0 open to 1 closed; if the grip
bounds are zero this robot has no hand you can actuate.
Reach is short. The torso origin is about 0.75 to 0.8 m above the floor, so a
table top at 0.7 m is just below z = 0, and a palm reaches only about 0.2 to
0.4 m in front of the torso. To touch something on a table, walk until your
body is nearly against the table edge (the edge stops your legs), then reach.
Something that still looks close in the camera may be out of reach. Each
palm stays on its own side of the body, so line a target up in front of the
hand you will use. Moving an arm can turn the body slightly; check base_pose.
Walk first and stop before reaching: hands are steadier when the body is
still. Side steps are slow for a humanoid: turn to face a target and walk
forward. A move can be larger than one step's bound; it is split into steps.
The body stops on its own a moment after commands stop. base_pose is
odometry from the reset pose and drifts, so trust the head camera."""


class G1MobileEmbodiment(G1Embodiment):
    """``g1_base`` plus both hands by palm position, in one action space."""

    NAME: ClassVar[str] = "g1_mobile"
    LABEL: ClassVar[str] = "G1"
    ENV_PREFIX: ClassVar[str] = "g1-mobile"
    CONFIG_CLS: ClassVar[type[LeggedConfig]] = G1MobileConfig
    DOCS: ClassVar[str] = _MOBILE_DOCS
    READY_PROMPT: ClassVar[str] = (
        "The G1 must already be standing in walking mode (set with the Unitree remote). "
        "Its arms will move to the front of its chest, then it will accept walking and "
        "arm commands. Clear the area around it, keep the remote in hand and a spotter "
        "or gantry ready, then press Enter..."
    )

    def __init__(self, config: LeggedConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._hand_targets = HOME_HANDS.copy()
        self._grip_targets = np.zeros(2, dtype=np.float64)
        #: Arms are only commanded between reset's homing and close's release.
        self._hands_active = False

    @property
    def _mobile_cfg(self) -> G1MobileConfig:
        assert isinstance(self._cfg, G1MobileConfig)
        return self._cfg

    @staticmethod
    def _hands(driver: LeggedDriver) -> HandsDriver:
        if not isinstance(driver, HandsDriver):
            raise TypeError(
                f"g1_mobile needs a backend that drives arms; got {type(driver).__name__}"
            )
        return driver

    # ------------------------------------------------------------------ hooks

    def _reset_extra(self, driver: LeggedDriver) -> None:
        """Take the arms over holding their measured pose, then glide the palms home."""
        hands = self._hands(driver)
        with self._lock:
            # The streamer stays off the arms until the takeover completes.
            self._hands_active = False
        cfg = self._mobile_cfg
        ticks = math.ceil((cfg.arm_weight_ramp_s + 1.0) * cfg.stream_hz)
        for _ in range(ticks):
            if hands.engage_hands():
                break
            self._sleep(1.0 / cfg.stream_hz)
        else:
            raise RuntimeError("arms did not finish taking over from the robot's controller")
        palms, _ = hands.read_hands()
        palms = np.asarray(palms, dtype=np.float64)
        with self._lock:
            # Start by holding exactly where the palms are, then take over.
            self._hand_targets = palms.copy()
            self._grip_targets = np.zeros(2, dtype=np.float64)
            self._hands_active = True
        step = cfg.hand_speed / cfg.control_hz
        distance = float(np.max(np.linalg.norm(HOME_HANDS - palms, axis=1)))
        count = max(1, math.ceil(distance / step))
        period = 1.0 / cfg.control_hz
        for index in range(1, count + 1):
            with self._lock:
                self._hand_targets = palms + (HOME_HANDS - palms) * (index / count)
                self._grip_targets = np.zeros(2, dtype=np.float64)
            if self._sim_paced:
                self.stream_once()
            self._sleep(period)
        with self._lock:
            self._hand_targets = HOME_HANDS.copy()

    def _apply_extra_locked(self, extra: Vec) -> None:
        """Move the palm targets and grips by this step's clamped displacement."""
        low, high = self._mobile_cfg.hand_box
        moved = self._hand_targets + extra[:6].reshape(2, 3)
        self._hand_targets = np.clip(moved, low, high)
        self._grip_targets = np.clip(self._grip_targets + extra[6:8], 0.0, 1.0)

    def _stream_extra(self, driver: LeggedDriver) -> None:
        """Send the current targets and adopt the reachable palms as targets."""
        hands = self._hands(driver)
        with self._lock:
            if not self._hands_active:
                return
            targets = self._hand_targets.copy()
            grips = self._grip_targets.copy()
        reached = np.asarray(hands.send_hands(targets, grips), dtype=np.float64)
        if reached.shape != (2, 3) or not np.all(np.isfinite(reached)):
            raise ValueError(f"backend returned invalid reachable palms: {reached!r}")
        with self._lock:
            # Only adopt the projection if no newer step moved the targets meanwhile.
            if np.array_equal(self._hand_targets, targets):
                self._hand_targets = reached

    def _extra_state(self, driver: LeggedDriver) -> dict[str, Vec]:
        """Measured palm positions and grips."""
        palms, grips = self._hands(driver).read_hands()
        palms = np.asarray(palms, dtype=np.float64)
        grips = np.asarray(grips, dtype=np.float64)
        if palms.shape != (2, 3) or grips.shape != (2,):
            raise ValueError("backend returned hand state of the wrong shape")
        if not (np.all(np.isfinite(palms)) and np.all(np.isfinite(grips))):
            raise ValueError("backend returned non-finite hand state")
        return {HANDS_KEY: palms.reshape(6), GRIPS_KEY: grips}

    def _release_extra(self, driver: LeggedDriver) -> None:
        """Hand the arms back to the robot's controller, ticking at stream rate."""
        with self._lock:
            active, self._hands_active = self._hands_active, False
        if not active:
            return
        hands = self._hands(driver)
        cfg = self._mobile_cfg
        ticks = math.ceil((cfg.arm_weight_ramp_s + 1.0) * cfg.stream_hz)
        for _ in range(ticks):
            if hands.release_hands():
                return
            self._wall_sleep(1.0 / cfg.stream_hz)
        raise RuntimeError("arms did not finish handing back to the robot's controller")
