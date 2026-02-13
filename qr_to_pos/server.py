import asyncio
import base64
import json
import signal
import time
from dataclasses import asdict

import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from qrdet import QRDetector
from websockets.asyncio.server import serve

from .processor import QRCode


class DetectionServer:
    """WebSocket server that receives images and returns QR detection results."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        model_size: str = "s",
        max_size: int = 16 * 1024 * 1024,
    ) -> None:
        self.host = host
        self.port = port
        self.max_size = max_size
        self.detector = QRDetector(model_size=model_size)

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

    async def handle(self, websocket) -> None:  # type: ignore
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    image = self.decode_image(message)
                elif isinstance(message, str):
                    payload = json.loads(message)
                    image_b64 = payload.get("image")
                    if image_b64 is None:
                        await websocket.send(
                            json.dumps({"error": "Missing 'image' field"})
                        )
                        continue
                    image = self.decode_image(base64.b64decode(image_b64))
                else:
                    await websocket.send(
                        json.dumps({"error": "Unsupported message type"})
                    )
                    continue

                start = time.perf_counter()
                qr_codes = self.detect(image)
                processing_time = time.perf_counter() - start

                response = {
                    "detections": [asdict(qr) for qr in qr_codes],
                    "count": len(qr_codes),
                    "processing_time": round(processing_time, 4),
                }
                await websocket.send(json.dumps(response))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
            except ValueError as e:
                await websocket.send(json.dumps({"error": str(e)}))
            except Exception as e:
                await websocket.send(json.dumps({"error": f"Processing error: {e}"}))

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        stop = loop.create_future()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set_result, None)
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
