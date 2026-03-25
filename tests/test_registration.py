from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marker_to_pos import registration


def _make_color_quad(points: np.ndarray) -> np.ndarray:
    image = np.full((400, 600, 3), 160, dtype=np.uint8)
    cv2.fillConvexPoly(image, points.astype(np.int32), (30, 30, 230))
    return image


def test_detect_box_corners_color_on_synthetic_quad():
    expected = np.array(
        [
            [120, 80],
            [480, 110],
            [450, 320],
            [100, 290],
        ],
        dtype=np.float32,
    )
    image = _make_color_quad(expected)

    corners = registration.detect_box_corners_color(
        image,
        min_area_ratio=0.05,
        canny_lo=50,
        canny_hi=150,
    )

    assert corners is not None
    np.testing.assert_allclose(corners, registration._order_corners(expected), atol=20)
