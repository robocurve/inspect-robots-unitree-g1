from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class FakeArm:
    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.joints = np.asarray(joints if joints is not None else np.zeros(14), dtype=float)
        self.body: Mapping[str, np.ndarray] = {
            "left_leg": np.arange(6, dtype=float),
            "right_leg": np.arange(6, 12, dtype=float),
            "waist": np.arange(12, 15, dtype=float),
        }
        self.publishes: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
        self.events: list[str] = []
        self.disconnect_error: BaseException | None = None

    def read_arm_joints(self) -> np.ndarray:
        return self.joints.copy()

    def read_body_state(self) -> Mapping[str, np.ndarray]:
        return self.body

    def publish_arm(self, q14: np.ndarray, weight: float, kp: np.ndarray, kd: np.ndarray) -> None:
        self.publishes.append((q14.copy(), weight, kp.copy(), kd.copy()))
        self.events.append(f"arm:{weight:.3f}")

    def disconnect(self) -> None:
        self.events.append("arm_disconnect")
        if self.disconnect_error is not None:
            raise self.disconnect_error


class FakeHand:
    def __init__(self, closure: np.ndarray | None = None) -> None:
        self.closure = np.asarray(closure if closure is not None else [0.25, 0.75], dtype=float)
        self.publishes: list[tuple[float, float]] = []
        self.events: list[str] = []
        self.disconnect_error: BaseException | None = None

    def read_closure(self) -> np.ndarray:
        return self.closure.copy()

    def publish_closure(self, left: float, right: float) -> None:
        self.publishes.append((left, right))
        self.events.append(f"hand:{left:.3f},{right:.3f}")

    def disconnect(self) -> None:
        self.events.append("hand_disconnect")
        if self.disconnect_error is not None:
            raise self.disconnect_error


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


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
        assert signum == self.SIGTERM
        return self.previous

    def signal(self, signum: int, handler: Any) -> None:
        self.handlers.append((signum, handler))
        self.previous = handler
