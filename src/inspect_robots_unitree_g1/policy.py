"""Isaac-GR00T ZMQ policy adapter for the canonical Unitree G1 contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_unitree_g1 import packing
from inspect_robots_unitree_g1.config import (
    ActionSpec,
    ConstSpec,
    Gr00tConfig,
    HandSpec,
    PackedSpec,
    SourceSpec,
    StateSourceSpec,
    action_box,
    observation_space,
)

WireObservation = Mapping[str, Any]
WireAction = Mapping[str, Any]
InferFn = Callable[[WireObservation], WireAction]


class SocketLike(Protocol):
    """Small pyzmq socket surface used by the reconnecting wrapper."""

    def setsockopt(self, option: int, value: int) -> None:
        """Set a socket option."""
        ...

    def connect(self, address: str) -> None:
        """Connect the REQ socket."""
        ...

    def send(self, payload: bytes) -> None:
        """Send one request."""
        ...

    def recv(self) -> bytes:
        """Receive one reply."""
        ...

    def close(self, linger: int = 0) -> None:
        """Close the socket without blocking."""
        ...


SocketFactory = Callable[[], SocketLike]


class ZmqReqTransport:
    """Testable REQ wrapper that recreates a socket after every timeout."""

    def __init__(
        self,
        socket_factory: SocketFactory,
        address: str,
        timeout_s: float,
        *,
        dumps: Callable[[Any], bytes],
        loads: Callable[[bytes], Any],
        timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,),
        rcvtimeo_option: int = 1,
        sndtimeo_option: int = 2,
    ) -> None:
        self._factory = socket_factory
        self._address = address
        self._timeout_ms = round(timeout_s * 1000)
        self._dumps = dumps
        self._loads = loads
        self._timeout_errors = timeout_errors
        self._rcvtimeo_option = rcvtimeo_option
        self._sndtimeo_option = sndtimeo_option
        self._socket = self._new_socket()

    def _new_socket(self) -> SocketLike:
        socket = self._factory()
        socket.setsockopt(self._rcvtimeo_option, self._timeout_ms)
        socket.setsockopt(self._sndtimeo_option, self._timeout_ms)
        socket.connect(self._address)
        return socket

    def get_action(self, observation: WireObservation) -> WireAction:
        """Send the GR00T envelope and unwrap the action from its list reply."""
        envelope = {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": {}},
        }
        try:
            self._socket.send(self._dumps(envelope))
            response = self._loads(self._socket.recv())
        except self._timeout_errors:
            self._socket.close(linger=0)
            self._socket = self._new_socket()
            raise
        if isinstance(response, Mapping) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        if not isinstance(response, list) or len(response) != 2:
            raise ValueError("GR00T reply must be a two-item [action, info] list")
        action = response[0]
        if not isinstance(action, Mapping):
            raise ValueError("GR00T reply action must be a mapping")
        return action

    def close(self) -> None:
        """Close the current REQ socket without lingering."""
        self._socket.close(linger=0)


def _default_infer(cfg: Gr00tConfig) -> InferFn:  # pragma: no cover - live transport
    """Build a pyzmq REQ client with msgpack-numpy serialization."""
    try:
        import msgpack
        import msgpack_numpy
        import zmq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The GR00T transport needs pyzmq, msgpack, and msgpack-numpy. Install with: "
            "pip install pyzmq msgpack msgpack-numpy",
            name=exc.name,
        ) from exc

    context = zmq.Context.instance()
    transport = ZmqReqTransport(
        lambda: context.socket(zmq.REQ),
        f"tcp://{cfg.host}:{cfg.port}",
        cfg.timeout_s,
        dumps=lambda value: msgpack.packb(value, default=msgpack_numpy.encode),
        loads=lambda value: msgpack.unpackb(value, object_hook=msgpack_numpy.decode, raw=False),
        timeout_errors=(zmq.error.Again,),
        rcvtimeo_option=zmq.RCVTIMEO,
        sndtimeo_option=zmq.SNDTIMEO,
    )
    return transport.get_action


class Gr00tPolicy:
    """Inspect Robots policy for an Isaac-GR00T PolicyServer."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "zmq": "pip install pyzmq",
        "msgpack": "pip install msgpack",
        "msgpack_numpy": "pip install msgpack-numpy",
    }

    def __init__(
        self,
        config: Gr00tConfig | None = None,
        *,
        infer_fn: InferFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else Gr00tConfig.from_kwargs(**flat)
        self._infer_fn = infer_fn
        self._clock = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(),
            control_hz=None,
        )
        self.config = PolicyConfig(
            action_horizon=self._cfg.action_horizon,
            replan_interval=self._cfg.replan_interval,
        )

    def reset(self, scene: Scene) -> None:
        """Store the language instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def _infer(self) -> InferFn:
        if self._infer_fn is None:
            self._infer_fn = _default_infer(self._cfg)
        return self._infer_fn

    def act(self, observation: Observation) -> ActionChunk:
        """Build nested wire inputs and return validated absolute 16-D actions."""
        if "head_cam" not in observation.images:
            raise ValueError("observation missing camera 'head_cam' required by gr00t")
        if packing.STATE_KEY not in observation.state:
            raise ValueError(f"observation missing state key {packing.STATE_KEY!r}")
        state = packing.validate_dim(observation.state[packing.STATE_KEY])
        if not np.all(np.isfinite(state)):
            raise ValueError(f"observation state {packing.STATE_KEY!r} contains non-finite values")
        required_passthrough = {
            spec[1] for spec in self._cfg.state_keys.values() if spec[0] == "state"
        }
        missing = sorted(required_passthrough - set(observation.state))
        if missing:
            raise ValueError(f"observation missing state keys required by gr00t: {missing}")

        _, image_name = self._cfg.image_key.split(".", 1)
        wire_state: dict[str, npt.NDArray[np.float32]] = {}
        for dotted, spec in self._cfg.state_keys.items():
            _, inner = dotted.split(".", 1)
            value = self._source_value(spec, state, observation)
            wire_state[inner] = np.asarray(value, dtype=np.float32)[None, None, :]
        instruction = observation.instruction or self._instruction or ""
        image = np.asarray(observation.images["head_cam"], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("head_cam must have shape (H, W, 3)")
        # Isaac-GR00T@9c7e746b2cd37a, gr00t/policy/policy.py requires nested
        # video/state/language groups with uint8 (B,T,H,W,C) video and float32
        # (B,T,D) state. Inner modality names are unprefixed.
        wire: dict[str, Any] = {
            "video": {image_name: image[None, None, ...]},
            "state": wire_state,
            "language": {"annotation.human.task_description": [[instruction]]},
        }

        started = self._clock()
        response = self._infer()(wire)
        elapsed = self._clock() - started
        adapted = self._assemble_actions(response)
        if self._cfg.actions_are_relative:
            adapted[:, :7] += state[:7]
            adapted[:, 8:15] += state[8:15]
        adapted = adapted[: self._cfg.action_horizon]
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in adapted],
            control_hz=self._cfg.control_hz,
            inference_latency_s=elapsed,
        )

    def _source_value(
        self, spec: SourceSpec, state: npt.NDArray[np.float64], observation: Observation
    ) -> npt.NDArray[np.float64]:
        tag = spec[0]
        if tag == "packed":
            packed = cast(PackedSpec, spec)
            out: npt.NDArray[np.float64] = np.asarray(
                state[packed[1] : packed[2]], dtype=np.float64
            ).copy()
            return out
        if tag == "state":
            state_spec = cast(StateSourceSpec, spec)
            values: npt.NDArray[np.float64] = np.asarray(
                observation.state[state_spec[1]], dtype=np.float64
            )
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError(f"state source {state_spec[1]!r} must be a finite 1-D array")
            return values
        if tag == "const":
            const = cast(ConstSpec, spec)
            return np.full(const[1], const[2], dtype=np.float64)
        hand = cast(HandSpec, spec)
        scalar = float(state[7 if hand[1] == "left" else 15])
        if self._cfg.hand_type == "dex1":
            return np.asarray([packing.dex1_scalar_to_stroke(scalar, self._cfg.dex1_stroke)])
        closed = np.asarray(
            packing.DEX3_LEFT_CLOSED_POSE if hand[1] == "left" else packing.DEX3_RIGHT_CLOSED_POSE,
            dtype=np.float64,
        )
        closed[0] = self._cfg.dex3_thumb_swing
        return packing.dex3_scalar_to_joints(scalar, packing.DEX3_OPEN_POSE, closed)

    def _assemble_actions(self, response: WireAction) -> npt.NDArray[np.float64]:
        arrays: list[tuple[ActionSpec, npt.NDArray[np.float64], str]] = []
        horizon: int | None = None
        for dotted, spec in self._cfg.action_keys.items():
            _, inner = dotted.split(".", 1)
            key = next(
                (
                    candidate
                    for candidate in (inner, dotted, f"action.{inner}")
                    if candidate in response
                ),
                None,
            )
            if key is None:
                raise ValueError(f"GR00T response missing action key {inner!r}")
            raw = np.asarray(response[key], dtype=np.float64)
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] == 0 or raw.shape[2] == 0:
                raise ValueError(f"GR00T action {key!r} has shape {raw.shape}; expected (1, T, D)")
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"GR00T action {key!r} contains non-finite values")
            if horizon is None:
                horizon = raw.shape[1]
            elif raw.shape[1] != horizon:
                raise ValueError("GR00T action keys returned inconsistent horizons")
            arrays.append((spec, raw[0], key))
        if horizon is None:
            raise ValueError("action_keys must not be empty")
        output = np.zeros((horizon, packing.TOTAL_DIM), dtype=np.float64)
        for spec, values, key in arrays:
            if spec[0] == "packed":
                width = spec[2] - spec[1]
                if values.shape[1] != width:
                    raise ValueError(
                        f"GR00T action {key!r} has width {values.shape[1]}; expected {width}"
                    )
                output[:, spec[1] : spec[2]] = values
            else:
                index = 7 if spec[1] == "left" else 15
                output[:, index] = self._compress_hand(values, spec[1])
        return output

    def _compress_hand(self, values: npt.NDArray[np.float64], side: str) -> npt.NDArray[np.float64]:
        if self._cfg.hand_type == "dex1":
            return np.asarray(
                [
                    packing.dex1_stroke_to_scalar(float(np.mean(row)), self._cfg.dex1_stroke)
                    for row in values
                ]
            )
        closed = np.asarray(
            packing.DEX3_LEFT_CLOSED_POSE if side == "left" else packing.DEX3_RIGHT_CLOSED_POSE,
            dtype=np.float64,
        )
        closed[0] = self._cfg.dex3_thumb_swing
        if values.shape[1] == 1:
            return np.clip(values[:, 0], 0.0, 1.0)
        if values.shape[1] != 7:
            raise ValueError(f"GR00T {side} hand action must have width 1 or 7")
        return np.asarray(
            [packing.dex3_joints_to_scalar(row, packing.DEX3_OPEN_POSE, closed) for row in values]
        )
