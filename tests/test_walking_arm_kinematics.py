"""G1 arm forward and inverse kinematics."""

import numpy as np
import pytest

from inspect_robots_unitree_g1.walking.arm_kinematics import (
    G1_ARMS,
    G1_LEFT_ARM,
    G1_RIGHT_ARM,
    fk,
    ik,
    jacobian,
)


def test_zero_pose_matches_the_urdf_and_arms_mirror() -> None:
    # Arms down, forearms forward: the palm sits ahead of the shoulder and below it.
    left = fk(G1_LEFT_ARM, np.zeros(7))
    right = fk(G1_RIGHT_ARM, np.zeros(7))
    np.testing.assert_allclose(left, (0.245, 0.152, 0.051), atol=2e-3)
    np.testing.assert_allclose(right * (1, -1, 1), left, atol=2e-3)
    assert G1_LEFT_ARM.dof == 7
    assert G1_LEFT_ARM.lower.shape == G1_LEFT_ARM.upper.shape == (7,)


def test_jacobian_matches_finite_differences() -> None:
    q = np.asarray(G1_LEFT_ARM.nominal) + 0.1
    eps = 1e-6
    numeric = np.stack(
        [
            (fk(G1_LEFT_ARM, q + eps * e) - fk(G1_LEFT_ARM, q - eps * e)) / (2 * eps)
            for e in np.eye(7)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jacobian(G1_LEFT_ARM, q), numeric, atol=1e-8)


@pytest.mark.parametrize("arm", G1_ARMS)
def test_ik_round_trips_reachable_poses(arm: object) -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        q = np.asarray(arm.nominal, dtype=np.float64)  # type: ignore[attr-defined]
        q[:4] = rng.uniform(arm.lower[:4] * 0.6, arm.upper[:4] * 0.6)  # type: ignore[attr-defined]
        solution, error = ik(arm, fk(arm, q), arm.nominal)  # type: ignore[arg-type,attr-defined]
        assert error < 2e-4
        assert np.all(solution >= arm.lower) and np.all(solution <= arm.upper)  # type: ignore[attr-defined]
        # The wrist holds its nominal angles.
        np.testing.assert_allclose(solution[4:], arm.nominal[4:])  # type: ignore[attr-defined]


def test_ik_tracks_small_steps_from_a_warm_start() -> None:
    q = np.asarray(G1_LEFT_ARM.nominal, dtype=np.float64)
    palm = fk(G1_LEFT_ARM, q)
    goal = np.asarray((0.30, 0.20, 0.10))
    while np.linalg.norm(goal - palm) > 1e-9:
        delta = goal - palm
        palm = palm + delta * min(1.0, 0.02 / float(np.linalg.norm(delta)))
        q, error = ik(G1_LEFT_ARM, palm, q, retry_error=1.0)
        assert error < 5e-3


def test_unreachable_target_returns_the_closest_pose() -> None:
    far = np.asarray((1.5, 0.2, 0.0))
    solution, error = ik(G1_LEFT_ARM, far, G1_LEFT_ARM.nominal)
    assert error > 0.9
    assert float(np.linalg.norm(fk(G1_LEFT_ARM, solution) - far)) == pytest.approx(error)


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        ik(G1_LEFT_ARM, (np.nan, 0.0, 0.0), G1_LEFT_ARM.nominal)
    with pytest.raises(ValueError, match="expected 7 joint angles"):
        fk(G1_LEFT_ARM, np.zeros(5))
