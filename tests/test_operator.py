import sys
from types import SimpleNamespace

import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_unitree_g1 import operator as module
from inspect_robots_unitree_g1.operator import OperatorIO


def test_wait_and_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_drain_stdin", lambda: None)
    prompts: list[str] = []
    operator = OperatorIO(input_fn=lambda prompt: prompts.append(prompt) or " YES ")
    operator.wait_ready("ready")
    assert operator.confirm_success() is True
    assert prompts == ["ready", "Did the robot succeed? [y/N]: "]
    assert OperatorIO(input_fn=lambda _: "no").confirm_success() is False


@pytest.mark.parametrize("error", [EOFError(), OSError()])
def test_wait_wraps_dead_stdin(error: BaseException) -> None:
    def fail(_: str) -> str:
        raise error

    with pytest.raises(EmbodimentFault, match="G1Config"):
        OperatorIO(input_fn=fail).wait_ready()


def test_drain_is_noop_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    module._drain_stdin()
