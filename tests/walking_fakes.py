"""Fakes for the walking tests: clock, driver, thread, atexit, signal and a wired rig."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from inspect_robots.scene import Scene

from inspect_robots_unitree_g1.walking.config import Go1Config, LeggedConfig
from inspect_robots_unitree_g1.walking.driver import Go1State
from inspect_robots_unitree_g1.walking.embodiment import Go1Embodiment
from inspect_robots_unitree_g1.walking.operator import OperatorIO


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


class FakeDriver:
    def __init__(self, height: int = 480, width: int = 640) -> None:
        self.pose = np.asarray((1.0, 2.0, 0.0))
        self.velocity = np.zeros(3)
        self.rpy = np.zeros(3)
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.events: list[Any] = []
        self.state_error: BaseException | None = None
        self.stand_error: BaseException | None = None
        self.disconnect_error: BaseException | None = None
        self.send_error: BaseException | None = None

    def send_velocity(self, forward: float, left: float, turn: float) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.events.append(("vel", round(forward, 6), round(left, 6), round(turn, 6)))

    def stand(self) -> None:
        self.events.append("stand")
        if self.stand_error is not None:
            raise self.stand_error

    def read_state(self) -> Go1State:
        if self.state_error is not None:
            raise self.state_error
        return Go1State(pose=self.pose.copy(), velocity=self.velocity.copy(), rpy=self.rpy.copy())

    def read_frame(self) -> np.ndarray:
        return self.frame

    def disconnect(self) -> None:
        self.events.append("disconnect")
        if self.disconnect_error is not None:
            raise self.disconnect_error


class ManualThread:
    """Records the streamer target instead of starting a thread."""

    def __init__(self, target: Callable[[], None]) -> None:
        self.target = target
        self.joined = False

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


class FakeAtexit:
    def __init__(self) -> None:
        self.registered: list[Any] = []
        self.unregistered: list[Any] = []

    def register(self, fn: Any) -> Any:
        self.registered.append(fn)
        return fn

    def unregister(self, fn: Any) -> None:
        self.unregistered.append(fn)


class FakeSignal:
    SIGTERM = 15

    def __init__(self, previous: Any = None) -> None:
        self.previous = previous
        self.handlers: list[tuple[int, Any]] = []

    def getsignal(self, signum: int) -> Any:
        return self.previous

    def signal(self, signum: int, handler: Any) -> None:
        self.handlers.append((signum, handler))
        self.previous = handler


class Rig:
    """One embodiment wired to fakes, with handles on each fake."""

    def __init__(
        self,
        cfg: LeggedConfig | None = None,
        *,
        embodiment_cls: type[Go1Embodiment] = Go1Embodiment,
        **overrides: Any,
    ) -> None:
        self.cfg = cfg or Go1Config(unattended=True)
        self.driver = FakeDriver(self.cfg.camera_height, self.cfg.camera_width)
        self.clock = FakeClock()
        self.threads: list[ManualThread] = []
        self.atexit = FakeAtexit()
        self.signal = FakeSignal()
        self.prompts: list[str] = []
        self.answer = ""

        def thread_factory(target: Callable[[], None]) -> ManualThread:
            thread = ManualThread(target)
            self.threads.append(thread)
            return thread

        def input_fn(prompt: str) -> str:
            self.prompts.append(prompt)
            return self.answer

        kwargs: dict[str, Any] = {
            "driver_factory": lambda _: self.driver,
            "operator": OperatorIO(input_fn=input_fn),
            "poll_end": lambda: False,
            "clock": self.clock,
            "sleep_fn": self.clock.sleep,
            "thread_factory": thread_factory,
            "atexit_module": self.atexit,
            "signal_module": self.signal,
        }
        kwargs.update(overrides)
        self.emb = embodiment_cls(self.cfg, **kwargs)

    def reset(self, instruction: str = "walk") -> Any:
        return self.emb.reset(Scene(id="s", instruction=instruction))
