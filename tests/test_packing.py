import numpy as np
import pytest

from inspect_robots_unitree_g1 import packing


def test_constants_pack_split_and_arm_slots() -> None:
    assert packing.ARM_DOF == 7
    assert packing.ARM_WIDTH == 8
    assert packing.TOTAL_DIM == 16
    assert packing.GRIPPER_IDXS == (7, 15)
    assert packing.DIM_LABELS[7] == "left_gripper"
    assert packing.DIM_LABELS[-1] == "right_gripper"
    left = np.arange(8)
    right = np.arange(10, 18)
    packed = packing.pack(left, right)
    actual_left, actual_right = packing.split(packed)
    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_right, right)
    np.testing.assert_array_equal(packing.arm_slots(packed), [*range(7), *range(10, 17)])
    actual_left[0] = 99
    assert packed[0] == 0


@pytest.mark.parametrize("value", [np.zeros((4, 4)), np.zeros(15)])
def test_validate_dim_rejects_wrong_shapes(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected a 16-D"):
        packing.validate_dim(value)


def test_dex1_conversion_clips_and_validates() -> None:
    assert packing.dex1_scalar_to_stroke(1.0) == 5.4
    assert packing.dex1_scalar_to_stroke(-1.0) == 0.0
    assert packing.dex1_stroke_to_scalar(2.7) == 0.5
    assert packing.dex1_stroke_to_scalar(99.0) == 1.0
    with pytest.raises(ValueError, match="stroke"):
        packing.dex1_stroke_to_scalar(0.0, 0.0)


def test_dex3_asymmetric_conversion_and_clipping() -> None:
    opened = np.asarray([0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 5.0])
    closed = np.asarray([0.0, -1.0, 3.0, 0.0, 1.0, 2.0, 7.0])
    middle = packing.dex3_scalar_to_joints(0.25, opened, closed)
    np.testing.assert_allclose(middle, closed + 0.25 * (opened - closed))
    assert packing.dex3_joints_to_scalar(middle, opened, closed) == pytest.approx(0.25)
    np.testing.assert_allclose(packing.dex3_scalar_to_joints(2.0, opened, closed), opened)
    with pytest.raises(ValueError, match="must differ"):
        packing.dex3_joints_to_scalar(opened, opened, opened)
