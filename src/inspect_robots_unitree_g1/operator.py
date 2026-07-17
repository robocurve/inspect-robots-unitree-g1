"""Injectable operator readiness and success confirmation for G1 runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inspect_robots.errors import EmbodimentFault

_AFFIRMATIVE = frozenset({"y", "yes", "1", "true", "success", "pass"})


@dataclass
class OperatorIO:
    """Console I/O isolated behind injectable callables."""

    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print

    def wait_ready(self, prompt: str = "Position the scene, then press Enter to start...") -> None:
        """Wait for readiness, converting dead stdin into a halting fault."""
        try:
            self.input_fn(prompt)
        except (EOFError, OSError) as exc:
            raise EmbodimentFault(
                "operator readiness prompt could not read stdin (no interactive terminal?). "
                "Run from a real TTY, inject an OperatorIO with a working input_fn, or set "
                "G1Config(unattended=True) (CLI: -E unattended=true)."
            ) from exc
        _drain_stdin()

    def confirm_success(self, prompt: str = "Did the robot succeed? [y/N]: ") -> bool:
        """Return whether the operator entered an affirmative verdict."""
        return self.input_fn(prompt).strip().lower() in _AFFIRMATIVE


def _drain_stdin() -> None:
    """Discard buffered TTY input so a stale newline cannot end step zero."""
    import sys

    if not sys.stdin.isatty():
        return
    import select  # pragma: no cover - TTY-bound

    while select.select([sys.stdin], [], [], 0)[0]:  # pragma: no cover - TTY-bound
        sys.stdin.readline()  # pragma: no cover - TTY-bound


def default_poll_end() -> bool:  # pragma: no cover - requires a real TTY
    """Return whether an operator pressed Enter without blocking."""
    import select
    import sys

    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True
