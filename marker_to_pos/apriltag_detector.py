"""Thin wrapper around pupil_apriltags.Detector with the project's shared interface.

The upstream C source for the AprilTag library is tracked as a git submodule at
extern/apriltag. At runtime, pupil-apriltags ships its own pre-compiled copy of
that library so no local build step is required.
"""
from __future__ import annotations

import cv2
import numpy as np
from pupil_apriltags import Detector as _PupilDetector

from .detection_geometry import bbox_xyxy_from_detection


class AprilTagDetector:
    """Wraps pupil_apriltags.Detector with the project's shared detect() interface.

    Each dict returned by detect() has the same keys consumed by
    DetectionServer.detect() and MarkerDetectionProcessor.process_frame():
      - quad_xy: (4, 2) float64 points in image space
      - bbox_xyxy: (x1, y1, x2, y2) derived from quad_xy
      - confidence: decision_margin float
          NOTE: decision_margin is a raw float (typically 0-200+)
          reflecting decode quality.
      - data: str(tag_id) — used as MarkerDetection.data in the wire protocol
      - _decoded: "family:tag_id" — injected for downstream display/identity
    """

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
    ) -> None:
        self._detector = _PupilDetector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
        )

    def detect(self, image: np.ndarray, *, is_bgr: bool = True) -> list[dict]:
        if is_bgr:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        results = []
        for r in self._detector.detect(gray):
            corners = r.corners  # (4, 2) float64, CCW order
            tag_id = int(r.tag_id)
            family = r.tag_family.decode("utf-8", errors="replace")
            detection = {
                "quad_xy": corners.copy(),
                "confidence": float(r.decision_margin),
                "data": str(tag_id),
                "_decoded": f"{family}:{tag_id}",
            }
            detection["bbox_xyxy"] = bbox_xyxy_from_detection(detection)
            results.append(detection)
        return results
