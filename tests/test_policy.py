from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_unitree_g1.config import Gr00tConfig
from inspect_robots_unitree_g1.policy import Gr00tPolicy, ZmqReqTransport


def observation() -> Observation:
    state = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.25, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0.75]
    )
    return Observation(
        images={"head_cam": np.arange(18, dtype=np.uint8).reshape(2, 3, 3)},
        state={
            "joint_pos": state,
            "left_leg": np.arange(6, dtype=float),
            "right_leg": np.arange(6, 12, dtype=float),
            "waist": np.arange(12, 15, dtype=float),
        },
        instruction="pick cube",
    )


def response(horizon: int = 3, *, prefixed: bool = False) -> dict[str, np.ndarray]:
    prefix = "action." if prefixed else ""
    return {
        f"{prefix}left_arm": np.arange(horizon * 7, dtype=float).reshape(1, horizon, 7),
        f"{prefix}right_arm": (100 + np.arange(horizon * 7, dtype=float)).reshape(1, horizon, 7),
        f"{prefix}left_hand": np.full((1, horizon, 1), 0.2),
        f"{prefix}right_hand": np.full((1, horizon, 1), 0.8),
        "waist": np.zeros((1, horizon, 3)),
    }


def test_nested_wire_observation_and_reassembly() -> None:
    captured: list[dict[str, Any]] = []
    times = iter([4.0, 4.25])

    def infer(wire: Any) -> Any:
        captured.append(wire)
        return response(prefixed=True)

    policy = Gr00tPolicy(infer_fn=infer, clock=lambda: next(times))
    policy.reset(Scene(id="x", instruction="fallback"))
    chunk = policy.act(observation())
    wire = captured[0]
    assert set(wire) == {"video", "state", "language"}
    assert set(wire["video"]) == {"ego_view"}
    assert wire["video"]["ego_view"].shape == (1, 1, 2, 3, 3)
    assert wire["video"]["ego_view"].dtype == np.uint8
    assert wire["state"]["left_arm"].shape == (1, 1, 7)
    assert wire["state"]["left_arm"].dtype == np.float32
    assert wire["state"]["left_hand"].shape == (1, 1, 7)
    assert wire["state"]["left_leg"].tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]
    assert wire["language"] == {"annotation.human.task_description": [["pick cube"]]}
    assert len(chunk) == 3
    assert chunk.control_hz == 10.0
    assert chunk.inference_latency_s == 0.25
    np.testing.assert_array_equal(chunk.actions[0].data[:7], np.arange(7))
    np.testing.assert_array_equal(chunk.actions[0].data[8:15], 100 + np.arange(7))
    assert chunk.actions[0].data[7] == 0.2
    assert chunk.actions[0].data[15] == 0.8
    assert policy.num_inferences == 1
    assert policy.config.action_horizon == 16
    assert not hasattr(policy.config, "host")


def test_dex1_templates_constants_truncation_and_instruction_fallback() -> None:
    config = Gr00tConfig(
        hand_type="dex1",
        action_horizon=2,
        state_keys={
            "state.left_arm": ("packed", 0, 7),
            "state.right_arm": ("packed", 8, 15),
            "state.left_hand": ("hand", "left"),
            "state.right_hand": ("hand", "right"),
            "state.fill": ("const", 2, 3.5),
        },
    )
    captured: list[Any] = []

    def infer(wire: Any) -> Any:
        captured.append(wire)
        out = response(4)
        out["left_hand"][:] = 2.7
        out["right_hand"][:] = 5.4
        return out

    policy = Gr00tPolicy(config, infer_fn=infer)
    policy.reset(Scene(id="x", instruction="scene prompt"))
    obs = observation()
    obs = Observation(images=obs.images, state=obs.state, instruction=None)
    chunk = policy.act(obs)
    assert len(chunk) == 2
    np.testing.assert_allclose(captured[0]["state"]["left_hand"], [[[1.35]]])
    assert captured[0]["state"]["fill"].tolist() == [[[3.5, 3.5]]]
    assert captured[0]["language"] == {"annotation.human.task_description": [["scene prompt"]]}
    assert chunk.actions[0].data[7] == 0.5
    assert chunk.actions[0].data[15] == 1.0


def test_relative_rows_use_same_anchor_and_leave_hands() -> None:
    rel = response(2)
    rel["left_arm"] = np.asarray([[[1.0] * 7, [2.0] * 7]])
    rel["right_arm"] = np.asarray([[[-1.0] * 7, [-2.0] * 7]])
    policy = Gr00tPolicy(actions_are_relative=True, infer_fn=lambda _: rel)
    chunk = policy.act(observation())
    np.testing.assert_allclose(chunk.actions[0].data[:7], np.arange(0.1, 0.8, 0.1) + 1)
    np.testing.assert_allclose(chunk.actions[1].data[:7], np.arange(0.1, 0.8, 0.1) + 2)
    np.testing.assert_allclose(chunk.actions[0].data[8:15], np.arange(1.1, 1.8, 0.1) - 1)
    assert chunk.actions[0].data[7] == 0.2
    assert chunk.actions[0].data[15] == 0.8


def test_default_is_absolute_passthrough() -> None:
    chunk = Gr00tPolicy(infer_fn=lambda _: response(1)).act(observation())
    np.testing.assert_array_equal(chunk.actions[0].data[:7], np.arange(7))


def test_dex3_seven_joint_action_compression() -> None:
    value = response(1)
    value["left_hand"] = np.asarray([[[0.0, 0.3, 0.4, -0.6, -0.7, -0.6, -0.7]]])
    value["right_hand"] = np.asarray([[[0.0, -0.3, -0.4, 0.6, 0.7, 0.6, 0.7]]])
    chunk = Gr00tPolicy(infer_fn=lambda _: value).act(observation())
    assert chunk.actions[0].data[7] == pytest.approx(0.5)
    assert chunk.actions[0].data[15] == pytest.approx(0.5)


def test_lazy_infer_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "inspect_robots_unitree_g1.policy._default_infer", lambda _: lambda __: response(1)
    )
    assert len(Gr00tPolicy().act(observation())) == 1


@pytest.mark.parametrize(
    ("obs", "message"),
    [
        (Observation(), "head_cam"),
        (Observation(images={"head_cam": np.zeros((1, 1, 3), dtype=np.uint8)}), "joint_pos"),
        (
            Observation(
                images={"head_cam": np.zeros((1, 1, 3), dtype=np.uint8)},
                state={"joint_pos": np.zeros(16)},
            ),
            "left_leg",
        ),
    ],
)
def test_required_observation_errors(obs: Observation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Gr00tPolicy(infer_fn=lambda _: response()).act(obs)


def test_template_runtime_errors() -> None:
    obs = observation()
    bad = observation()
    bad_state = dict(bad.state)
    bad_state["left_leg"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="finite 1-D"):
        Gr00tPolicy(infer_fn=lambda _: response()).act(
            Observation(images=bad.images, state=bad_state)
        )
    bad_state = dict(obs.state)
    bad_state["joint_pos"] = np.full(16, np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        Gr00tPolicy(infer_fn=lambda _: response()).act(
            Observation(images=obs.images, state=bad_state)
        )
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        Gr00tPolicy(infer_fn=lambda _: response()).act(
            Observation(images={"head_cam": np.zeros((2, 2))}, state=obs.state)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("left_arm"), "missing action key"),
        (lambda value: value.__setitem__("left_arm", np.zeros((2, 3, 7))), "expected"),
        (lambda value: value.__setitem__("left_arm", np.full((1, 3, 7), np.nan)), "non-finite"),
        (lambda value: value.__setitem__("right_arm", np.zeros((1, 2, 7))), "inconsistent"),
        (lambda value: value.__setitem__("left_arm", np.zeros((1, 3, 6))), "expected 7"),
        (lambda value: value.__setitem__("left_hand", np.zeros((1, 3, 2))), "width 1 or 7"),
    ],
)
def test_action_response_errors(mutate: Any, message: str) -> None:
    value = response()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        Gr00tPolicy(infer_fn=lambda _: value).act(observation())


def test_empty_action_template_errors() -> None:
    cfg = Gr00tConfig(action_keys={})
    with pytest.raises(ValueError, match="must not be empty"):
        Gr00tPolicy(cfg, infer_fn=lambda _: {}).act(observation())


class Socket:
    def __init__(self, reply: bytes | BaseException) -> None:
        self.reply = reply
        self.options: list[tuple[int, int]] = []
        self.addresses: list[str] = []
        self.sent: list[bytes] = []
        self.closed: list[int] = []

    def setsockopt(self, option: int, value: int) -> None:
        self.options.append((option, value))

    def connect(self, address: str) -> None:
        self.addresses.append(address)

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply

    def close(self, linger: int = 0) -> None:
        self.closed.append(linger)


def test_req_transport_envelope_close_and_timeout_recreate() -> None:
    first = Socket(TimeoutError())
    second = Socket(b"ok")
    sockets = iter([first, second])
    decoded: list[Any] = []

    def dumps(value: Any) -> bytes:
        decoded.append(value)
        return b"request"

    transport = ZmqReqTransport(
        lambda: next(sockets),
        "tcp://host:5555",
        1.5,
        dumps=dumps,
        loads=lambda _: [{"left_arm": []}, {}],
        rcvtimeo_option=10,
        sndtimeo_option=11,
    )
    with pytest.raises(TimeoutError):
        transport.get_action({"video": {}})
    assert first.closed == [0]
    assert second.options == [(10, 1500), (11, 1500)]
    assert second.addresses == ["tcp://host:5555"]
    action = transport.get_action({"video": {}})
    assert action == {"left_arm": []}
    assert decoded[0] == {
        "endpoint": "get_action",
        "data": {"observation": {"video": {}}, "options": {}},
    }
    transport.close()
    assert second.closed == [0]


@pytest.mark.parametrize("reply", [[{}], [1, {}]])
def test_req_transport_reply_validation(reply: Any) -> None:
    socket = Socket(b"ok")
    transport = ZmqReqTransport(lambda: socket, "x", 1, dumps=lambda _: b"", loads=lambda _: reply)
    with pytest.raises(ValueError, match="reply"):
        transport.get_action({})


def test_req_transport_surfaces_server_error_reply() -> None:
    socket = Socket(b"ok")
    transport = ZmqReqTransport(
        lambda: socket,
        "x",
        1,
        dumps=lambda _: b"",
        loads=lambda _: {"error": "boom in the handler"},
    )
    with pytest.raises(RuntimeError, match="GR00T server error: boom in the handler"):
        transport.get_action({})
