"""A full ``eval()`` over the kinematic backend with a scripted walking policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from inspect_robots import eval as robots_eval
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_unitree_g1.walking.config import LeggedConfig, action_box, observation_space
from inspect_robots_unitree_g1.walking.driver import KinematicLeggedDriver
from inspect_robots_unitree_g1.walking.embodiment import LeggedEmbodiment
from walking_fakes import FakeAtexit, FakeClock, FakeSignal, ManualThread


class WalkForward:
    """Emits full-speed forward steps, the way the agent splits a move_by."""

    def __init__(self) -> None:
        self.info = PolicyInfo(
            name="walk-forward",
            action_space=action_box(),
            observation_space=observation_space(),
        )
        self.config = PolicyConfig(action_horizon=5)

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.asarray((0.05, 0.0, 0.0)))] * 5)


def test_eval_walks_the_kinematic_robot_forward(tmp_path: Path) -> None:
    cfg = LeggedConfig(unattended=True, camera_height=48, camera_width=64)
    clock = FakeClock()
    driver = KinematicLeggedDriver(cfg, clock=clock, response_time_s=0.0)
    emb: LeggedEmbodiment

    def sleep_while_streaming(duration: float) -> None:
        # Stand-in for the streamer thread: one send per control period.
        emb.stream_once()
        clock.sleep(duration)

    emb = LeggedEmbodiment(
        cfg,
        driver_factory=lambda _: driver,
        clock=clock,
        sleep_fn=sleep_while_streaming,
        thread_factory=ManualThread,
        atexit_module=FakeAtexit(),
        signal_module=FakeSignal(),
    )
    task = Task(
        name="walk",
        scenes=[Scene(id="w", instruction="walk forward")],
        scorer=episode_length(),
        max_steps=20,
    )
    logs = robots_eval(task, WalkForward(), emb, sinks=[], seed=0, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error
    assert emb.num_steps == 20
    # 20 steps at 0.5 m/s for 0.1 s each is one metre.
    np.testing.assert_allclose(driver.read_state().pose[:2], (1.0, 0.0), atol=1e-6)
    # eval() parked the robot after the trial (standing, still connected).
    emb.stream_once()
    clock.sleep(1.0)
    np.testing.assert_allclose(driver.read_state().velocity, 0.0)
    # An embodiment passed in by object is the caller's to close.
    assert driver.connected is True
    emb.close()
    assert driver.connected is False
