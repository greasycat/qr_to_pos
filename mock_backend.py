import asyncio
import json
import math
import random
import time

import websockets


def make_detection(marker_id: str, x_pct: float, y_pct: float) -> dict:
    x_pct = max(0.0, min(100.0, x_pct))
    y_pct = max(0.0, min(100.0, y_pct))
    cx = int(x_pct * 6.4)  # fake 640-wide image
    cy = int(y_pct * 4.8)  # fake 480-tall image
    h = 20
    return {
        "data": marker_id,
        "bbox": [cx - h, cy - h, cx + h, cy + h],
        "confidence": 0.99,
        "decoded": f"tag36h11:{marker_id}",
        "depth_centroid": [float(cx), float(cy)],
        "depth_centroid_pct": [x_pct, y_pct],
    }


def get_detections() -> list:
    t = time.time()

    # Pattern 1 — marker "1": jiggle around fixed centre (30, 30)
    jx = 30.0 + random.gauss(0, 1.5)
    jy = 30.0 + random.gauss(0, 1.5)

    # Pattern 2 — marker "2": x sweeps 10→90 (80% range, well above 60% threshold)
    mx = 50.0 + 40.0 * math.sin(t * 0.6)
    my = 50.0

    # Pattern 3 — markers "3" & "4": orbit (65, 65) at radius 20, opposite phases
    ocx, ocy, r = 65.0, 65.0, 20.0
    angle = t * 0.8
    r3x = ocx + r * math.cos(angle)
    r3y = ocy + r * math.sin(angle)
    r4x = ocx + r * math.cos(angle + math.pi)
    r4y = ocy + r * math.sin(angle + math.pi)

    return [
        make_detection("1", jx, jy),
        make_detection("2", mx, my),
        make_detection("3", r3x, r3y),
        make_detection("4", r4x, r4y),
    ]


async def handle(websocket):
    async for message in websocket:
        action = "detect"
        try:
            if isinstance(message, str):
                payload = json.loads(message)
                action = payload.get("action", "detect")
        except Exception:
            pass

        if action not in ("detect", "detect_unity"):
            continue

        dets = get_detections()
        response = {
            "action": action,
            "homography": None,
            "count": len(dets),
            "processing_time": 0.001,
            "detections": dets,
            "error": None,
        }
        await websocket.send(json.dumps(response))


async def main():
    print("Mock backend running on ws://localhost:8765")
    print("  marker 1 — jiggle   (fixed ~30,30 + noise)")
    print("  marker 2 — movement (x sweeps 10→90)")
    print("  marker 3 — rotation (orbit centre 65,65)")
    print("  marker 4 — rotation (orbit 180° opposite marker 3)")
    async with websockets.serve(handle, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
