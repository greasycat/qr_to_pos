import asyncio
import base64
from datetime import datetime, UTC
import json
import shutil
import signal
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from websockets.asyncio.server import serve

from .apriltag_detector import AprilTagDetector
from .detection_geometry import bbox_xyxy_from_detection
from .processor import QRCode
from .registration import (
    compute_homography,
    detect_box_corners_color,
    detect_box_corners_depth,
    load_registration,
    save_registration,
    transform_points,
    transform_bbox_to_depth,
)

_DEFAULT_REGISTRATION_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "registration" / "homography.npy"
)
_DEFAULT_REGISTRATION_COORDS_PATH = _DEFAULT_REGISTRATION_PATH.with_name("coords.yml")
_NORMALIZED_COORD_EPSILON = 1e-4
_DEFAULT_DEBUG_CAPTURE_DIR = Path(tempfile.gettempdir()) / "marker_to_pos" / "ws_debug"
_DEFAULT_MAX_SAVED_IMAGES = 200
_DEFAULT_UNITY_IMAGE_ACTIONS = ("flip_h",)


class DetectionServer:
    """WebSocket server that receives images and returns AprilTag detection results."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        model_size: str = "s",
        max_size: int = 16 * 1024 * 1024,
        registration_path: str | Path = _DEFAULT_REGISTRATION_PATH,
        registration_coords_path: str | Path | None = None,
        save_decoding_images: bool | None = None,
        debug_capture_dir: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.max_size = max_size
        self.registration_path = Path(registration_path)
        if registration_coords_path is None:
            registration_coords_path = self.registration_path.with_name(_DEFAULT_REGISTRATION_COORDS_PATH.name)
        self.registration_coords_path = Path(registration_coords_path)
        config = self._load_server_config()
        detector_config = config.get("detector", {}) if isinstance(config.get("detector"), dict) else {}
        detector_type = self._coerce_detector_type(detector_config.get("type"))
        self.detector = AprilTagDetector()
        debug_config = config.get("ws_debug", {}) if isinstance(config.get("ws_debug"), dict) else {}
        processing_config = (
            config.get("ws_processing", {}) if isinstance(config.get("ws_processing"), dict) else {}
        )
        self.save_decoding_images = (
            self._coerce_bool(debug_config.get("save_decoding_images", False))
            if save_decoding_images is None
            else save_decoding_images
        )
        self.max_saved_images = self._coerce_positive_int(
            debug_config.get("max_saved_images"),
            _DEFAULT_MAX_SAVED_IMAGES,
        )
        self.debug_capture_dir = (
            Path(debug_capture_dir)
            if debug_capture_dir is not None
            else _DEFAULT_DEBUG_CAPTURE_DIR
        )
        self.unity_image_actions = self._coerce_image_actions(
            processing_config.get("unity_image_actions"),
            default=_DEFAULT_UNITY_IMAGE_ACTIONS,
        )
        self._capture_sequence = 0
        self.detector_type = detector_type

    def detect(self, image: np.ndarray) -> list[QRCode]:
        detections = self.detector.detect(image=image, is_bgr=True)
        if not detections:
            return []

        qr_codes: list[QRCode] = []
        for detection in detections:
            x1, y1, x2, y2 = bbox_xyxy_from_detection(detection)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            decoded = detection.get("_decoded") or detection.get("decoded")

            qr_codes.append(
                QRCode(
                    data=detection.get("data", ""),  # type: ignore
                    bbox=(x1, y1, x2, y2),
                    confidence=detection.get("confidence", 1.0),  # type: ignore
                    decoded=decoded,
                )
            )
        return qr_codes

    def decode_image(self, raw: bytes) -> np.ndarray:
        buf = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image

    def decode_base64_image(self, image_b64: str) -> np.ndarray:
        return self.decode_image(base64.b64decode(image_b64))

    def apply_image_actions(
        self,
        image: np.ndarray,
        *,
        actions: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        processed = image
        for action in actions:
            if action == "flip_h":
                processed = cv2.flip(processed, 1)
                continue
            if action == "flip_v":
                processed = cv2.flip(processed, 0)
                continue
            if action == "r_180_plus":
                processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
                continue
            if action == "r_180_minus":
                processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
                continue
            raise ValueError(f"Unsupported image action: {action}")

        return processed

    def decode_depth_text(self, raw_text: str) -> np.ndarray:
        lines = raw_text.splitlines()
        if not lines:
            raise ValueError("Depth TXT is empty")

        data_lines = [line for line in lines[1:] if line.strip()]
        if not data_lines:
            raise ValueError("Depth TXT has no depth rows")

        rows = [list(map(float, line.strip().split("\t"))) for line in data_lines]
        return np.array(rows, dtype=np.float32)

    def _serialize_points(self, pts: np.ndarray | None) -> list[list[float]] | None:
        if pts is None:
            return None
        return [[float(x), float(y)] for x, y in pts]

    def _serialize_point(self, pt: np.ndarray | None) -> list[float] | None:
        if pt is None:
            return None
        return [float(pt[0]), float(pt[1])]

    def _serialize_matrix(self, matrix: np.ndarray | None) -> list[list[float]] | None:
        if matrix is None:
            return None
        return [[float(value) for value in row] for row in matrix]

    def _load_server_config(self) -> dict[str, Any]:
        config_path = self.registration_path.with_name("config.yml")
        if not config_path.exists():
            return {}

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _coerce_detector_type(self, value: Any) -> str:
        if value is None:
            return "apriltag"
        if not isinstance(value, str):
            raise ValueError("detector.type must be a string when configured")

        detector_type = value.strip().lower()
        if detector_type == "apriltag":
            return detector_type

        raise ValueError(
            f"Unsupported detector type: {detector_type}. This build only supports 'apriltag'."
        )

    def _coerce_positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, parsed)

    def _coerce_image_actions(
        self,
        value: Any,
        *,
        default: tuple[str, ...],
    ) -> tuple[str, ...]:
        if value is None:
            actions = list(default)
        elif isinstance(value, list):
            actions = value
        else:
            raise ValueError("ws_processing.unity_image_actions must be a list of action names")

        normalized = []
        for action in actions:
            if not isinstance(action, str):
                raise ValueError("unity image action names must be strings")
            action_name = action.strip()
            if action_name not in {"flip_h", "flip_v", "r_180_plus", "r_180_minus"}:
                raise ValueError(f"Unsupported unity image action: {action_name}")
            normalized.append(action_name)
        return tuple(normalized)

    def _load_registration_matrix(self) -> np.ndarray | None:
        if not self.registration_path.exists():
            return None
        return load_registration(str(self.registration_path))

    def _load_registration_depth_corners(self) -> np.ndarray | None:
        if not self.registration_coords_path.exists():
            return None

        with self.registration_coords_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        depth_corners = payload.get("depth_corners")
        if not isinstance(depth_corners, list) or len(depth_corners) != 4:
            return None

        try:
            points = np.asarray(depth_corners, dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if points.shape != (4, 2):
            return None
        return points

    def _compute_polygon_centroid(self, pts: np.ndarray) -> np.ndarray | None:
        if pts.size == 0:
            return None

        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = float(np.sum(cross) * 0.5)

        if np.isclose(signed_area, 0.0):
            return np.mean(pts, axis=0)

        centroid_x = float(np.sum((x + x_next) * cross) / (6.0 * signed_area))
        centroid_y = float(np.sum((y + y_next) * cross) / (6.0 * signed_area))
        return np.array([centroid_x, centroid_y], dtype=np.float64)

    def _compute_depth_centroid_pct(
        self,
        depth_bbox: np.ndarray | None,
        depth_corners: np.ndarray | None,
    ) -> np.ndarray | None:
        if depth_bbox is None or depth_corners is None:
            return None

        centroid = self._compute_polygon_centroid(depth_bbox)
        if centroid is None:
            return None

        rect_space = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        depth_to_rect = compute_homography(depth_corners.astype(np.float32), rect_space)
        normalized = transform_points(np.array([centroid], dtype=np.float64), depth_to_rect)[0]
        if not np.all(np.isfinite(normalized)):
            return None
        if np.any(normalized < -_NORMALIZED_COORD_EPSILON) or np.any(
            normalized > 1.0 + _NORMALIZED_COORD_EPSILON
        ):
            return None

        normalized = np.clip(normalized, 0.0, 1.0)
        return np.round(normalized * 100.0, 4)

    def _serialize_detection(
        self,
        qr: QRCode,
        homography: np.ndarray | None,
        depth_corners: np.ndarray | None,
    ) -> dict[str, Any]:
        detection = asdict(qr)
        serialized_homography = self._serialize_matrix(homography)
        detection["homography"] = serialized_homography
        detection["depth_bbox"] = None
        detection["depth_centroid"] = None
        detection["depth_centroid_pct"] = None
        if homography is not None and qr.bbox is not None:
            depth_bbox = transform_bbox_to_depth(list(qr.bbox), homography)
            depth_centroid = self._compute_polygon_centroid(depth_bbox)
            detection["depth_bbox"] = self._serialize_points(depth_bbox)
            detection["depth_centroid"] = self._serialize_point(depth_centroid)
            detection["depth_centroid_pct"] = self._serialize_point(
                self._compute_depth_centroid_pct(depth_bbox, depth_corners)
            )
        return detection

    def _iter_debug_capture_dirs(self) -> list[Path]:
        if not self.debug_capture_dir.exists():
            return []
        return sorted(path for path in self.debug_capture_dir.iterdir() if path.is_dir())

    def _prune_debug_captures(self) -> None:
        capture_dirs = self._iter_debug_capture_dirs()
        excess = len(capture_dirs) - self.max_saved_images
        if excess <= 0:
            return

        for path in capture_dirs[:excess]:
            shutil.rmtree(path, ignore_errors=True)

    def _save_debug_capture(
        self,
        image: np.ndarray,
        response: dict[str, Any],
        request_source: str,
        request_action: str,
    ) -> Path | None:
        if not self.save_decoding_images:
            return None

        try:
            self.debug_capture_dir.mkdir(parents=True, exist_ok=True)
            self._capture_sequence += 1
            timestamp = datetime.now(UTC)
            capture_dir = self.debug_capture_dir / (
                f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{self._capture_sequence:06d}"
            )
            capture_dir.mkdir(parents=False, exist_ok=False)

            image_path = capture_dir / "input.png"
            if not cv2.imwrite(str(image_path), image):
                raise ValueError(f"Failed to write debug image to {image_path}")

            processed_image_path = capture_dir / "processed.png"
            self._save_processed_detection_image(
                image=image,
                detections=response.get("detections"),
                output_path=processed_image_path,
            )

            response_path = capture_dir / "response.json"
            with response_path.open("w", encoding="utf-8") as handle:
                json.dump(response, handle, indent=2, sort_keys=True)

            metadata = {
                "saved_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "request_source": request_source,
                "request_action": request_action,
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "count": response.get("count"),
                "processing_time": response.get("processing_time"),
            }
            metadata_path = capture_dir / "metadata.yml"
            with metadata_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(metadata, handle, sort_keys=False)

            self._prune_debug_captures()
            return capture_dir
        except Exception as exc:
            print(f"DetectionServer: Failed to save debug capture: {exc}")
            return None

    def _save_processed_detection_image(
        self,
        image: np.ndarray,
        detections: Any,
        output_path: Path,
    ) -> None:
        processed = image.copy()

        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, dict):
                    continue

                bbox = detection.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                try:
                    x1, y1, x2, y2 = [int(value) for value in bbox]
                except (TypeError, ValueError):
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                cv2.ellipse(
                    processed,
                    center,
                    axes,
                    0,
                    0,
                    360,
                    (0, 255, 0),
                    2,
                )

        if not cv2.imwrite(str(output_path), processed):
            raise ValueError(f"Failed to write processed image to {output_path}")

    def process_detect_request(
        self,
        image: np.ndarray,
        *,
        request_source: str,
        action: str = "detect",
        image_actions: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        processed_image = self.apply_image_actions(
            image,
            actions=image_actions,
        )
        response = self.detect_response(processed_image, action=action)
        self._save_debug_capture(
            processed_image,
            response,
            request_source=request_source,
            request_action=action,
        )
        return response

    def detect_response(self, image: np.ndarray, action: str = "detect") -> dict[str, Any]:
        start = time.perf_counter()
        qr_codes = self.detect(image)
        processing_time = time.perf_counter() - start
        homography = self._load_registration_matrix()
        depth_corners = self._load_registration_depth_corners()
        return {
            "action": action,
            "homography": self._serialize_matrix(homography),
            "detections": [self._serialize_detection(qr, homography, depth_corners) for qr in qr_codes],
            "count": len(qr_codes),
            "processing_time": round(processing_time, 4),
        }

    def update_corners_response(self, color_image: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
        color_pts = detect_box_corners_color(color_image)
        depth_pts = detect_box_corners_depth(depth)
        return {
            "action": "update_corners",
            "color_corners": self._serialize_points(color_pts),
            "depth_corners": self._serialize_points(depth_pts),
            "color_detected": color_pts is not None,
            "depth_detected": depth_pts is not None,
        }

    def update_registration_response(
        self,
        color_corners: list[list[float]] | np.ndarray,
        depth_corners: list[list[float]] | np.ndarray,
        save: bool = True,
    ) -> dict[str, Any]:
        color_pts = np.array(color_corners, dtype=np.float32)
        depth_pts = np.array(depth_corners, dtype=np.float32)
        if color_pts.shape != (4, 2) or depth_pts.shape != (4, 2):
            raise ValueError("Expected color_corners and depth_corners to be 4x2 point arrays")

        H = compute_homography(color_pts, depth_pts)
        saved_path = None
        if save:
            self.registration_path.parent.mkdir(parents=True, exist_ok=True)
            save_registration(H, str(self.registration_path))
            saved_path = str(self.registration_path)

        return {
            "action": "update_registration",
            "homography": H.tolist(),
            "saved_path": saved_path,
        }

    def handle_json_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if action is None and "image" in payload:
            action = "detect"

        if action == "detect":
            print("Detect request receieved")
            image_b64 = payload.get("image")
            if image_b64 is None:
                raise ValueError("Missing 'image' field")
            image = self.decode_base64_image(image_b64)
            return self.process_detect_request(
                image,
                request_source="calibration",
                action="detect",
                image_actions=(),
            )

        if action == "detect_unity":
            print("Unity detect request receieved")
            image_b64 = payload.get("image")
            if image_b64 is None:
                raise ValueError("Missing 'image' field")
            image = self.decode_base64_image(image_b64)
            return self.process_detect_request(
                image,
                request_source="unity",
                action="detect_unity",
                image_actions=self.unity_image_actions,
            )

        if action == "update_corners":
            print("Corner update request receieved")
            image_b64 = payload.get("color_image")
            depth_text = payload.get("depth_text")
            if image_b64 is None:
                raise ValueError("Missing 'color_image' field")
            if depth_text is None:
                raise ValueError("Missing 'depth_text' field")
            color_image = self.decode_base64_image(image_b64)
            depth = self.decode_depth_text(depth_text)
            return self.update_corners_response(color_image, depth)

        if action == "update_registration":
            color_corners = payload.get("color_corners")
            depth_corners = payload.get("depth_corners")
            if color_corners is None:
                raise ValueError("Missing 'color_corners' field")
            if depth_corners is None:
                raise ValueError("Missing 'depth_corners' field")
            return self.update_registration_response(
                color_corners=color_corners,
                depth_corners=depth_corners,
                save=bool(payload.get("save", True)),
            )

        raise ValueError(f"Unsupported action: {action}")

    async def handle(self, websocket) -> None:  # type: ignore
        async for message in websocket:
            action = "detect" if isinstance(message, bytes) else None
            try:
                if isinstance(message, bytes):
                    response = self.process_detect_request(
                        self.decode_image(message),
                        request_source="binary",
                        action="detect",
                        image_actions=(),
                    )
                elif isinstance(message, str):
                    payload = json.loads(message)
                    action = payload.get("action")
                    response = self.handle_json_message(payload)
                else:
                    await websocket.send(
                        json.dumps({"error": "Unsupported message type"})
                    )
                    continue

                await websocket.send(json.dumps(response))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except ValueError as e:
                await websocket.send(json.dumps({"error": str(e), "action": action}))
            except Exception as e:
                await websocket.send(
                    json.dumps({"error": f"Processing error: {e}", "action": action})
                )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        stop = loop.create_future()

        def _set_stop() -> None:
            if not stop.done():
                stop.set_result(None)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _set_stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        async with serve(self.handle, self.host, self.port, max_size=self.max_size) as server:
            print(f"Detection server listening on ws://{self.host}:{self.port}")
            try:
                await stop
            except NotImplementedError:
                # Fallback for Windows: just run forever
                await asyncio.Future()

        print("Server stopped.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AprilTag detection WebSocket server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--save-decoding-images",
        action="store_true",
        default=None,
        help="Save each detect request image and response metadata into a temp debug folder",
    )
    args = parser.parse_args()

    server = DetectionServer(
        host=args.host,
        port=args.port,
        save_decoding_images=args.save_decoding_images,
    )
    if server.save_decoding_images:
        print(
            "Debug capture enabled: saving detect requests to "
            f"{server.debug_capture_dir} (max_saved_images={server.max_saved_images})"
        )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
