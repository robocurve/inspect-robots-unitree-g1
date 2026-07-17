import builtins

import pytest

from inspect_robots_unitree_g1 import _unitree


def test_loader_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name == "unitree_sdk2py":
            raise ModuleNotFoundError(name="unitree_sdk2py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        _unitree._load_unitree()
    text = str(caught.value)
    assert "git+https://github.com/unitreerobotics/unitree_sdk2_python" in text
    assert "CYCLONEDDS_HOME" in text
    assert "L2 adjacency" in text


def test_loader_reraises_unrelated_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name == "unitree_sdk2py":
            raise ModuleNotFoundError(name="dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        _unitree._load_unitree()
    assert caught.value.name == "dependency"


def test_loader_returns_present_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    real_import = builtins.__import__

    def present(name: str, *args: object, **kwargs: object) -> object:
        if name == "unitree_sdk2py":
            return sentinel
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", present)
    assert _unitree._load_unitree() is sentinel
