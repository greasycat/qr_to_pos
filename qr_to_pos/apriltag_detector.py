"""Thin wrapper around pupil_apriltags.Detector with a qrdet-compatible interface.

The upstream C source for the AprilTag library is tracked as a git submodule at
extern/apriltag. At runtime, pupil-apriltags ships its own pre-compiled copy of
that library so no local build step is required.
"""
from __future__ import annotations

import cv2
import numpy as np
from pupil_apriltags import Detector as _PupilDetector


class AprilTagDetector:
    """Wraps pupil_apriltags.Detector with a qrdet-compatible detect() interface.

    Each dict returned by detect() has the same keys consumed by
    DetectionServer.detect() and QRCodeProcessor.process_frame():
      - bbox_xyxy: (x1, y1, x2, y2) derived from the four tag corners
      - confidence: decision_margin float
          NOTE: decision_margin is NOT on the [0,1] scale used by qrdet.
          It is a raw float (typically 0-200+) reflecting decode quality.
          Code that thresholds on confidence values should account for this.
      - data: str(tag_id) — used as QRCode.data in the wire protocol
      - _decoded: "family:tag_id" — injected so server.detect() can skip pyzbar
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
            x1 = float(corners[:, 0].min())
            y1 = float(corners[:, 1].min())
            x2 = float(corners[:, 0].max())
            y2 = float(corners[:, 1].max())
            tag_id = int(r.tag_id)
            family = r.tag_family.decode("utf-8", errors="replace")
            results.append({
                "bbox_xyxy": (x1, y1, x2, y2),
                "confidence": float(r.decision_margin),
                "data": str(tag_id),
                "_decoded": f"{family}:{tag_id}",
            })
        return results
