from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _coerce_quad_points(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        return None
    return points


def bbox_xyxy_from_detection(detection: Mapping[str, Any]) -> tuple[float, float, float, float]:
    quad = _coerce_quad_points(detection.get("quad_xy"))
    if quad is not None:
        return (
            float(quad[:, 0].min()),
            float(quad[:, 1].min()),
            float(quad[:, 0].max()),
            float(quad[:, 1].max()),
        )

    bbox = detection.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise KeyError("Detection is missing a valid bbox_xyxy field")

    return tuple(float(value) for value in bbox)
