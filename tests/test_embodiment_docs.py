from inspect_robots_unitree_g1.config import DEFAULT_JOINT_HIGH, DEFAULT_JOINT_LOW, G1Config
from inspect_robots_unitree_g1.embodiment import _DOCS, G1Embodiment
from inspect_robots_unitree_g1.packing import DIM_LABELS


def test_docs_name_every_dimension_once_without_bounds() -> None:
    docs = G1Embodiment().info.docs or ""
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1
    for value in (*DEFAULT_JOINT_LOW, *DEFAULT_JOINT_HIGH):
        assert str(value) not in docs


def test_docs_extra_append_semantics() -> None:
    assert G1Embodiment(G1Config(docs_extra="  rig note\n")).info.docs == _DOCS + "\n\nrig note"
    assert G1Embodiment(G1Config(docs_extra=" \n ")).info.docs == _DOCS
