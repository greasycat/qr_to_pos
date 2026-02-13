import json
from pathlib import Path

import pytest
import websockets
from websockets.asyncio.server import serve

from qr_to_pos.server import DetectionServer

IMAGE_PATH = Path(__file__).resolve().parent.parent / "qrs.png"


@pytest.fixture()
async def server_url():
    server = DetectionServer(host="localhost", port=0, model_size="s")
    async with serve(server.handle, server.host, 0, max_size=server.max_size) as ws_server:
        # Get the actual port assigned by the OS
        port = ws_server.sockets[0].getsockname()[1]
        yield f"ws://localhost:{port}"


@pytest.mark.asyncio
async def test_detect_qrs_from_image(server_url):
    image_bytes = IMAGE_PATH.read_bytes()

    async with websockets.connect(server_url) as ws:
        await ws.send(image_bytes)
        raw = await ws.recv()

    result = json.loads(raw)

    # Should not be an error response
    assert "error" not in result

    # Should have detected QR codes
    assert result["count"] > 0
    assert len(result["detections"]) == result["count"]
    assert isinstance(result["processing_time"], float)

    for det in result["detections"]:
        # Every detection must have a bounding box with 4 ints
        assert det["bbox"] is not None
        assert len(det["bbox"]) == 4
        assert all(isinstance(c, int) for c in det["bbox"])

        # Confidence should be a positive number
        assert det["confidence"] is not None
        assert det["confidence"] > 0

        # pyzbar should have decoded at least some of them
    decoded_values = [d["decoded"] for d in result["detections"] if d["decoded"]]
    assert len(decoded_values) > 0, "pyzbar should decode at least one QR code"

    print(decoded_values)