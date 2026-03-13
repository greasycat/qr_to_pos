import asyncio
import base64
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pyzbar.pyzbar import decode as pyzbar_decode
from qrdet import QRDetector
from websockets.asyncio.server import serve

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


class DetectionServer:
    """WebSocket server that receives images and returns QR detection results."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        model_size: str = "s",
        max_size: int = 16 * 1024 * 1024,
        registration_path: str | Path = _DEFAULT_REGISTRATION_PATH,
        registration_coords_path: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.max_size = max_size
        self.detector = QRDetector(model_size=model_size)
        self.registration_path = Path(registration_path)
        if registration_coords_path is None:
            registration_coords_path = self.registration_path.with_name(_DEFAULT_REGISTRATION_COORDS_PATH.name)
        self.registration_coords_path = Path(registration_coords_path)

    def detect(self, image: np.ndarray) -> list[QRCode]:
        detections = self.detector.detect(image=image, is_bgr=True)
        if not detections:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        qr_codes = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox_xyxy"]  # type: ignore
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Decode QR content from the cropped region using pyzbar
            pad = 10
            h, w = gray.shape
            crop = gray[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]
            decoded = None
            results = pyzbar_decode(crop)
            if results:
                decoded = results[0].data.decode("utf-8", errors="replace")

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

    def detect_response(self, image: np.ndarray) -> dict[str, Any]:
        start = time.perf_counter()
        qr_codes = self.detect(image)
        processing_time = time.perf_counter() - start
        homography = self._load_registration_matrix()
        depth_corners = self._load_registration_depth_corners()
        return {
            "action": "detect",
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
            return self.detect_response(image)

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
                    response = self.detect_response(self.decode_image(message))
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

    parser = argparse.ArgumentParser(description="QR detection WebSocket server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model-size",
        default="s",
        choices=["n", "s", "m", "l"],
        help="YOLO model size for QR detection",
    )
    args = parser.parse_args()

    server = DetectionServer(
        host=args.host, port=args.port, model_size=args.model_size
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
