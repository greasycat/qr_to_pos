import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .apriltag_detector import AprilTagDetector
from .camera import Camera, Frame
from .detection_geometry import bbox_xyxy_from_detection


@dataclass
class ProcessingResult:
    """Container for processing results with metadata."""
    result: list["QRCode"]
    frame_index: int
    frame_timestamp: float
    processing_time: float


@dataclass
class QRCode:
    """Detected marker data."""
    data: str
    bbox: tuple[int, int, int, int] | None = None
    confidence: float | None = None
    decoded: str | None = None


class QRCodeProcessor:
    """Detects AprilTags in frames."""

    def __init__(
        self,
        camera: Camera,
        min_interval: float = 0.1,
        model_size: str = "s",
        detector_type: str = "apriltag",
    ) -> None:
        self.camera = camera
        self.min_interval = min_interval
        self.model_size = model_size
        self.detector_type = self._coerce_detector_type(detector_type)
        self.detector = AprilTagDetector()
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._callbacks: list[Callable[[ProcessingResult], None]] = []
        self._callback_lock = threading.Lock()
        self._last_processed_index = -1

    @staticmethod
    def _coerce_detector_type(value: Any) -> str:
        if value is None:
            return "apriltag"
        if not isinstance(value, str):
            raise ValueError("detector_type must be a string when provided")

        detector_type = value.strip().lower()
        if detector_type == "apriltag":
            return detector_type

        raise ValueError(
            f"Unsupported detector_type: {detector_type}. This build only supports 'apriltag'."
        )

    def on_result(self, callback: Callable[[ProcessingResult], None]) -> None:
        with self._callback_lock:
            self._callbacks.append(callback)

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running:
            return

        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=timeout)

    def _process_loop(self) -> None:
        while self._running:
            try:
                frame = self.camera.get_latest_frame()

                if frame is None or frame.index <= self._last_processed_index:
                    time.sleep(self.min_interval)
                    continue

                # Process the frame
                start_time = time.time()
                result = self.process_frame(frame)
                processing_time = time.time() - start_time

                self._last_processed_index = frame.index

                if result is not None and len(result) > 0:
                    processing_result = ProcessingResult(
                        result=result,
                        frame_index=frame.index,
                        frame_timestamp=frame.timestamp,
                        processing_time=processing_time,
                    )
                    self._emit_result(processing_result)

                # Rate limiting
                time.sleep(self.min_interval)

            except Exception as e:
                if self._running:
                    print(f"Error in processing loop: {e}")
                    time.sleep(self.min_interval)

    def _emit_result(self, result: ProcessingResult) -> None:
        with self._callback_lock:
            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception as e:
                    print(f"Error in result callback: {e}")

    def process_frame(self, frame: Frame) -> list[QRCode] | None:
        try:
            detections = self.detector.detect(image=frame.data, is_bgr=True)

            if not detections:
                return None

            # Convert detections to QRCode objects
            qr_codes = []

            for detection in detections:
                x1, y1, x2, y2 = bbox_xyxy_from_detection(detection)
                confidence = detection.get("confidence", 1.0)
                data = detection.get("data", "")

                # Ensure coordinates are standard Python ints
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                qr_code = QRCode(
                    data=data,  # type: ignore
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,  # type: ignore
                    decoded=detection.get("_decoded") or detection.get("decoded"),
                )
                qr_codes.append(qr_code)

            # Return all detected markers
            return qr_codes if qr_codes else None

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def __enter__(self) -> "QRCodeProcessor":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
