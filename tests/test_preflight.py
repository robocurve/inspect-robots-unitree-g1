import json

import pytest
from inspect_robots.compat import CompatibilityReport, CompatIssue

from inspect_robots_unitree_g1 import preflight


def report(severity: str | None = None) -> CompatibilityReport:
    issues = [] if severity is None else [CompatIssue(severity, "code", "detail")]
    return CompatibilityReport(issues=issues)


def test_build_and_preflight_with_task_and_injection() -> None:
    policy, embodiment = preflight.build()
    assert policy.info.name == "gr00t"
    assert embodiment.info.name == "g1_arms"
    assert preflight.run_preflight().issues == []
    assert preflight.run_preflight("cubepick-reach", policy=policy, embodiment=embodiment).ok
    sentinel = report("warning")
    assert preflight.run_preflight(check=lambda *_args, **_kwargs: sentinel) is sentinel


@pytest.mark.parametrize(
    ("value", "args", "code", "text"),
    [
        (report(), [], 0, "OK:"),
        (report("warning"), [], 0, "WARNING"),
        (report("error"), [], 1, "INCOMPATIBLE"),
        (report(), ["--dry-run"], 0, "dry-run"),
    ],
)
def test_human_cli(
    value: CompatibilityReport,
    args: list[str],
    code: int,
    text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert preflight.main(args, run=lambda *_args, **_kwargs: value) == code
    assert text in capsys.readouterr().out


def test_json_includes_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        preflight.main(["--json", "--dry-run"], run=lambda *_args, **_kwargs: report("error")) == 1
    )
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "dry_run": True,
        "errors": [{"code": "code", "message": "detail"}],
        "warnings": [],
    }


def test_main_default_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert preflight.main([]) == 0
    assert "OK:" in capsys.readouterr().out
