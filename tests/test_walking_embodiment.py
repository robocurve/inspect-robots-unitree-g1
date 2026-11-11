from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_unitree_g1.walking.config import LeggedConfig
from inspect_robots_unitree_g1.walking.driver import LeggedState
from inspect_robots_unitree_g1.walking.embodiment import LeggedEmbodiment
from walking_fakes import FakeDriver, Rig


@pytest.fixture
def rig() -> Rig:
    return Rig()


# ---------------------------------------------------------------- reset


def test_reset_connects_stands_and_zeroes_odometry(rig: Rig) -> None:
    rig.driver.pose = np.asarray((5.0, -1.0, 0.7))
    obs = rig.reset("go to the door")
    assert rig.driver.events == ["stand"]
    np.testing.assert_allclose(obs.state["base_pose"], (0.0, 0.0, 0.0))
    assert obs.instruction == "go to the door"
    assert obs.images["front_cam"].shape == (480, 640, 3)
    assert len(rig.threads) == 1
    assert rig.atexit.registered == [rig.emb.close]
    assert rig.signal.handlers[0][0] == rig.signal.SIGTERM
    # A second reset reuses the connection and streamer.
    rig.reset()
    assert len(rig.threads) == 1


def test_reset_prompts_operator_when_attended() -> None:
    rig = Rig(LeggedConfig())
    rig.reset()
    assert rig.prompts and "Clear the area" in rig.prompts[0]


def test_reset_rejects_non_callable_camera_reader() -> None:
    rig = Rig(camera_reader="not callable")
    with pytest.raises(ValueError, match="camera_reader must be callable"):
        rig.reset()


def test_reset_closes_when_first_observation_fails() -> None:
    rig = Rig(camera_reader=lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="camera frame shape"):
        rig.reset()
    assert rig.driver.events[-1] == "disconnect"


def test_injected_camera_reader_wins_over_driver_frame() -> None:
    frame = np.full((480, 640, 3), 7, dtype=np.uint8)
    rig = Rig(camera_reader=lambda: frame)
    assert int(rig.reset().images["front_cam"][0, 0, 0]) == 7


# ---------------------------------------------------------------- step


def test_step_clamps_paces_and_hands_velocity_to_streamer(rig: Rig) -> None:
    rig.reset()
    result = rig.emb.step(Action(data=np.asarray((1.0, -1.0, 0.05))))
    assert rig.clock.sleeps == [pytest.approx(0.1)]
    assert result.terminated is False
    assert rig.emb.num_steps == 1
    # Before the deadline the streamer sends the clamped velocity.
    rig.emb.stream_once()
    assert rig.driver.events[-1] == ("vel", 0.5, -0.3, 0.5)


def test_watchdog_stands_after_deadline(rig: Rig) -> None:
    rig.reset()
    rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    rig.clock.now += 0.09  # inside the 0.1 s grace after the control period
    rig.emb.stream_once()
    assert rig.driver.events[-1] == ("vel", 0.5, 0.0, 0.0)
    rig.clock.now += 0.02
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"


def test_streamer_stands_before_any_step(rig: Rig) -> None:
    rig.emb.stream_once()  # no driver yet: nothing to do
    assert rig.driver.events == []
    rig.reset()
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros(2), "must have 3 values"),
        (np.asarray((0.0, math.nan, 0.0)), "only finite values"),
    ],
)
def test_step_rejects_bad_actions(rig: Rig, data: Any, message: str) -> None:
    rig.reset()
    with pytest.raises(ValueError, match=message):
        rig.emb.step(Action(data=data))


def test_step_before_reset_raises(rig: Rig) -> None:
    with pytest.raises(RuntimeError, match="before reset"):
        rig.emb.step(Action(data=np.zeros(3)))


def test_step_failure_closes_the_robot(rig: Rig) -> None:
    rig.reset()
    rig.driver.state_error = RuntimeError("udp lost")
    with pytest.raises(RuntimeError, match="udp lost"):
        rig.emb.step(Action(data=np.zeros(3)))
    assert rig.driver.events[-2:] == ["stand", "disconnect"]


def test_geofence_ends_trial_and_stands(rig: Rig) -> None:
    rig.reset()
    rig.driver.pose = rig.driver.pose + np.asarray((3.5, 0.0, 0.0))
    result = rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    assert result.terminated is True
    assert result.termination_reason == "geofence"
    assert result.info["distance_m"] == pytest.approx(3.5)
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"


def test_geofence_can_be_disabled() -> None:
    rig = Rig(LeggedConfig(unattended=True, geofence_m=None))
    rig.reset()
    rig.driver.pose = rig.driver.pose + np.asarray((50.0, 0.0, 0.0))
    assert rig.emb.step(Action(data=np.zeros(3))).terminated is False


@pytest.mark.parametrize(("answer", "reason"), [("y", "success"), ("n", "failure")])
def test_operator_end_scores_and_stands(answer: str, reason: str) -> None:
    rig = Rig(LeggedConfig(), poll_end=lambda: True)
    rig.reset()
    rig.answer = answer
    result = rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    assert result.terminated is True
    assert result.termination_reason == reason
    assert result.info["operator_confirmed"] is (reason == "success")
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"


def test_pose_is_relative_to_reset_heading(rig: Rig) -> None:
    rig.driver.pose = np.asarray((0.0, 0.0, math.pi / 2))
    rig.reset()
    # Moving +y in the odometry frame is "forward" for a robot facing +y.
    rig.driver.pose = np.asarray((0.0, 1.0, math.pi))
    obs = rig.emb.step(Action(data=np.zeros(3))).observation
    np.testing.assert_allclose(obs.state["base_pose"], (1.0, 0.0, math.pi / 2), atol=1e-9)


@pytest.mark.parametrize("field", ["pose", "velocity", "rpy"])
def test_invalid_backend_state_is_refused(rig: Rig, field: str) -> None:
    rig.reset()
    bad = {"pose": rig.driver.pose, "velocity": rig.driver.velocity, "rpy": rig.driver.rpy}
    bad[field] = np.asarray((0.0, math.inf, 0.0))

    def read_state() -> LeggedState:
        return LeggedState(**bad)

    rig.driver.read_state = read_state  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=f"invalid {field}"):
        rig.emb.step(Action(data=np.zeros(3)))


# ---------------------------------------------------------------- streamer thread


def test_stream_loop_runs_until_stopped(rig: Rig) -> None:
    rig.reset()
    calls: list[int] = []
    original = rig.emb.stream_once

    def counting() -> None:
        calls.append(1)
        original()
        if len(calls) == 3:
            rig.emb._stop_streamer.set()

    rig.emb.stream_once = counting  # type: ignore[method-assign]
    rig.threads[0].target()
    assert len(calls) == 3


def test_stream_failure_is_reported_and_stops_the_next_step(rig: Rig) -> None:
    rig.reset()
    rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    rig.driver.send_error = OSError("socket closed")
    rig.threads[0].target()  # loop exits after recording the error
    report = rig.emb.health()
    assert report.ok is False
    assert "socket closed" in report.summary()
    with pytest.raises(RuntimeError, match="streamer stopped"):
        rig.emb.step(Action(data=np.zeros(3)))
    assert rig.driver.events[-1] == "disconnect"


def test_stream_failure_during_step_is_raised_after_observing(rig: Rig) -> None:
    rig.reset()

    def failing_sleep(duration: float) -> None:
        rig.emb._stream_error = OSError("dropped mid-step")

    rig.emb._sleep = failing_sleep
    with pytest.raises(RuntimeError, match="dropped mid-step"):
        rig.emb.step(Action(data=np.zeros(3)))


# ---------------------------------------------------------------- health, park, close


def test_health_before_and_after_connect(rig: Rig) -> None:
    assert rig.emb.health().checks == ()
    rig.reset()
    assert rig.emb.health().ok is True
    rig.driver.state_error = RuntimeError("no state")
    report = rig.emb.health()
    assert report.ok is False
    assert "no state" in report.summary()


def test_park_stands_without_disconnecting(rig: Rig) -> None:
    rig.emb.park()  # not connected: no-op
    assert rig.driver.events == []
    rig.reset()
    rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    rig.emb.park()
    assert rig.driver.events[-1] == "stand"
    rig.emb.stream_once()
    assert rig.driver.events[-1] == "stand"
    assert "disconnect" not in rig.driver.events


def test_close_stands_joins_and_disconnects_once(rig: Rig) -> None:
    rig.reset()
    rig.emb.close()
    assert rig.driver.events[-2:] == ["stand", "disconnect"]
    assert rig.threads[0].joined is True
    assert rig.atexit.unregistered == [rig.emb.close]
    rig.emb.close()
    assert rig.driver.events.count("disconnect") == 1
    with pytest.raises(RuntimeError, match="after close"):
        rig.emb.step(Action(data=np.zeros(3)))


def test_close_tolerates_streamer_without_join() -> None:
    rig = Rig(thread_factory=lambda target: object())
    rig.reset()
    rig.emb.close()
    assert rig.driver.events[-1] == "disconnect"


def test_close_disconnects_even_when_stand_fails_and_keeps_backstops(rig: Rig) -> None:
    rig.reset()
    rig.driver.stand_error = RuntimeError("stand failed")
    with pytest.raises(RuntimeError, match="stand failed"):
        rig.emb.close()
    assert rig.driver.events[-1] == "disconnect"
    assert rig.atexit.unregistered == []


def test_close_reports_disconnect_failure(rig: Rig) -> None:
    rig.reset()
    rig.driver.disconnect_error = OSError("busy")
    with pytest.raises(OSError, match="busy"):
        rig.emb.close()


def test_del_is_best_effort(rig: Rig) -> None:
    rig.reset()
    rig.driver.disconnect_error = OSError("busy")
    rig.emb.__del__()  # swallowed
    assert rig.driver.events[-1] == "disconnect"


# ---------------------------------------------------------------- connect and backstops


def test_thread_start_failure_releases_the_driver() -> None:
    def broken(target: Any) -> Any:
        raise RuntimeError("no threads")

    rig = Rig(thread_factory=broken)
    with pytest.raises(RuntimeError, match="no threads"):
        rig.reset()
    assert rig.driver.events == ["disconnect"]
    assert rig.atexit.unregistered == [rig.emb.close]


def test_signal_registration_failure_unwinds_atexit() -> None:
    rig = Rig()

    def broken_signal(signum: int, handler: Any) -> None:
        raise ValueError("not main thread")

    rig.signal.signal = broken_signal  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="not main thread"):
        rig.reset()
    assert rig.atexit.unregistered == [rig.emb.close]
    assert rig.driver.events == ["disconnect"]


def test_reconnect_after_close_reregisters_backstops(rig: Rig) -> None:
    rig.reset()
    rig.emb.close()
    rig.reset()
    assert len(rig.threads) == 2
    assert rig.atexit.registered == [rig.emb.close, rig.emb.close]


def test_sigterm_closes_then_chains_previous_handler() -> None:
    seen: list[int] = []
    rig = Rig()
    rig.signal.previous = lambda signum, frame: seen.append(signum)
    rig.reset()
    handler = rig.signal.handlers[0][1]
    handler(15, None)
    assert rig.driver.events[-1] == "disconnect"
    assert seen == [15]


def test_sigterm_with_the_default_handler_still_terminates() -> None:
    rig = Rig()
    rig.signal.previous = rig.signal.SIG_DFL
    rig.reset()
    rig.signal.handlers[0][1](15, None)
    assert rig.driver.events[-1] == "disconnect"
    assert rig.signal.raised == [15]  # default action restored and re-raised
    assert rig.signal.handlers[-1] == (15, rig.signal.SIG_DFL)


def test_sigterm_with_an_ignoring_handler_only_cleans_up() -> None:
    rig = Rig()
    rig.signal.previous = rig.signal.SIG_IGN
    rig.reset()
    rig.signal.handlers[0][1](15, None)
    assert rig.driver.events[-1] == "disconnect"
    assert rig.signal.raised == []


def test_unregister_is_idempotent(rig: Rig) -> None:
    rig.emb._unregister_backstops()
    assert rig.atexit.unregistered == []


def test_register_is_idempotent(rig: Rig) -> None:
    rig.reset()
    rig.emb._register_backstops()
    assert rig.atexit.registered == [rig.emb.close]


def test_default_scene_instruction_flows_through() -> None:
    driver = FakeDriver()
    emb = LeggedEmbodiment(
        LeggedConfig(unattended=True),
        driver_factory=lambda _: driver,
        thread_factory=lambda target: None,
        clock=lambda: 0.0,
        sleep_fn=lambda _: None,
        atexit_module=Rig().atexit,
        signal_module=Rig().signal,
    )
    assert emb.reset(Scene(id="x", instruction="sit")).instruction == "sit"
    emb.close()


class SimPacedDriver(FakeDriver):
    """A FakeDriver with its own clock, like the Isaac Lab backend."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0
        self.advances: list[tuple[float, Any]] = []

    def sim_time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        last = self.events[-1] if self.events else None
        self.advances.append((seconds, last))
        self.now += seconds


def test_sim_paced_driver_drives_the_clock_and_latches_before_physics() -> None:
    driver = SimPacedDriver()
    rig = Rig(driver_factory=lambda _: driver)
    rig.reset()
    rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    # Physics ran one control period, with the new command already latched.
    assert driver.advances == [(pytest.approx(0.1), ("vel", 0.5, 0.0, 0.0))]
    assert rig.clock.sleeps == []
    # Wall time passing (an LLM thinking) does not expire the command in sim time.
    rig.clock.now += 60.0
    rig.emb.stream_once()
    assert driver.events[-1] == ("vel", 0.5, 0.0, 0.0)
    # Sim time passing does.
    driver.now += 1.0
    rig.emb.stream_once()
    assert driver.events[-1] == "stand"
    rig.emb.close()
    # The wall clock is back for the next connection.
    rig.driver.pose = np.zeros(3)
    rig.emb._driver_factory = lambda _: rig.driver
    rig.reset()
    rig.emb.step(Action(data=np.asarray((0.05, 0.0, 0.0))))
    assert rig.clock.sleeps == [pytest.approx(0.1)]
    rig.emb.close()


def test_sim_pacing_is_undone_when_connecting_fails() -> None:
    driver = SimPacedDriver()

    def broken_thread(target: Any) -> Any:
        raise RuntimeError("no threads")

    rig = Rig(driver_factory=lambda _: driver, thread_factory=broken_thread)
    with pytest.raises(RuntimeError, match="no threads"):
        rig.reset()
    assert rig.emb._clock is rig.clock
