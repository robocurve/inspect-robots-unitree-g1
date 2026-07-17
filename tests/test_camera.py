import numpy as np
import pytest

from inspect_robots_unitree_g1.embodiment import decode_jpeg_frame


class Cv2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 2

    def __init__(self, decoded: np.ndarray | None) -> None:
        self.decoded = decoded

    def imdecode(self, encoded: np.ndarray, mode: int) -> np.ndarray | None:
        assert encoded.dtype == np.uint8
        assert mode == self.IMREAD_COLOR
        return self.decoded

    def cvtColor(self, image: np.ndarray, mode: int) -> np.ndarray:
        assert mode == self.COLOR_BGR2RGB
        return image[..., ::-1]


def test_decode_jpeg_to_rgb() -> None:
    bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_jpeg_frame(lambda: b"jpeg", Cv2(bgr)), [[[3, 2, 1]]])


@pytest.mark.parametrize("payload", [b"", None])
def test_empty_payload(payload: bytes | None) -> None:
    with pytest.raises(RuntimeError, match="empty or stale"):
        decode_jpeg_frame(lambda: payload, Cv2(np.zeros((1, 1, 3))))  # type: ignore[arg-type,return-value]


def test_timeout_and_invalid_decode() -> None:
    def timeout() -> bytes:
        raise TimeoutError

    with pytest.raises(RuntimeError, match="timed out"):
        decode_jpeg_frame(timeout, Cv2(np.zeros((1, 1, 3))))
    with pytest.raises(RuntimeError, match="invalid JPEG"):
        decode_jpeg_frame(lambda: b"bad", Cv2(None))
