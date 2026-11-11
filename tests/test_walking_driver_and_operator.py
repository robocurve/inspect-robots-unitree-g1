import math

import numpy as np
import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_unitree_g1.walking.config import Go1Config
from inspect_robots_unitree_g1.walking.driver import Go1Driver, KinematicGo1Driver, render_top_down
from inspect_robots_unitree_g1.walking.operator import OperatorIO
from walking_fakes import FakeClock


def make(tau: float = 0.0) -> tuple[KinematicGo1Driver, FakeClock]:
    clock = FakeClock()
    driver = KinematicGo1Driver(
        Go1Config(camera_height=48, camera_width=64), clock=clock, response_time_s=tau
    )
    return driver, clock


def test_kinematic_driver_satisfies_protocol() -> None:
    driver, _ = make()
    assert isinstance(driver, Go1Driver)


def test_walks_forward_then_turns() -> None:
    driver, clock = make()
    driver.send_velocity(0.5, 0.0, 0.0)
    clock.now += 2.0
    state = driver.read_state()
    np.testing.assert_allclose(state.pose, (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(state.velocity, (0.5, 0.0, 0.0))
    driver.send_velocity(0.0, 0.0, math.pi / 2)
    clock.now += 1.0
    assert driver.read_state().pose[2] == pytest.approx(math.pi / 2)
    driver.send_velocity(0.5, 0.0, 0.0)
    clock.now += 2.0
    np.testing.assert_allclose(driver.read_state().pose[:2], (1.0, 1.0), atol=1e-9)


def test_lag_and_stand() -> None:
    driver, clock = make(tau=0.15)
    driver.send_velocity(0.5, 0.0, 0.0)
    clock.now += 0.15
    assert 0.2 < driver.read_state().velocity[0] < 0.5
    driver.stand()
    clock.now += 2.0
    assert abs(driver.read_state().velocity[0]) < 1e-3
    assert driver.read_state().rpy[0] == 0.0  # zero elapsed time: no change


def test_frame_moves_with_the_robot() -> None:
    driver, clock = make()
    first = driver.read_frame()
    assert first.shape == (48, 64, 3)
    assert first.dtype == np.uint8
    driver.send_velocity(0.5, 0.0, 0.0)
    clock.now += 0.3
    assert not np.array_equal(first, driver.read_frame())
    driver.disconnect()
    assert driver.connected is False


def test_render_marks_origin_and_heading() -> None:
    frame = render_top_down(np.zeros(3), 100, 100)
    assert tuple(frame[50, 50]) == (40, 120, 220)  # origin dot under the robot
    assert tuple(frame[30, 50]) == (200, 40, 40)  # heading bar toward the top


def test_operator_prompts() -> None:
    answers = iter(["", "Yes"])
    io = OperatorIO(input_fn=lambda prompt: next(answers))
    io.wait_ready()
    assert io.confirm_success() is True

    def dead(prompt: str) -> str:
        raise EOFError

    with pytest.raises(EmbodimentFault, match="unattended=true"):
        OperatorIO(input_fn=dead).wait_ready()
