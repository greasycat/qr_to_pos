import base64
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qr_to_pos.server import DetectionServer

IMAGE_PATH = Path(__file__).resolve().parent.parent / "assets" / "fake_background_multiple_qr.png"


class _DummyDetector:
    def detect(self, image, is_bgr=True):
        return [
            {
                "bbox_xyxy": [12, 18, 96, 104],
                "confidence": 0.97,
                "data": "dummy",
            }
        ]


class _DummyDecode:
    def __init__(self, data: bytes) -> None:
        self.data = data


@pytest.fixture()
def stubbed_server(monkeypatch):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("qr_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])
    return DetectionServer(host="localhost", port=0, model_size="s")


def test_detect_qrs_from_image(stubbed_server):
    image_bytes = IMAGE_PATH.read_bytes()
    image = stubbed_server.decode_image(image_bytes)
    result = stubbed_server.detect_response(image)

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


def test_update_corners_action(monkeypatch):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    server = DetectionServer(host="localhost", port=0, model_size="s")

    monkeypatch.setattr(
        "qr_to_pos.server.detect_box_corners_color",
        lambda _image: np.array([[10, 10], [110, 12], [108, 90], [12, 88]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "qr_to_pos.server.detect_box_corners_depth",
        lambda _depth: np.array([[8, 14], [95, 11], [101, 80], [6, 83]], dtype=np.float32),
    )

    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = server.handle_json_message(
        {
            "action": "update_corners",
            "color_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "depth_text": "Frame size: 2 x 2\n1.0\t1.1\n1.2\t1.3\n",
        }
    )

    assert response["action"] == "update_corners"
    assert response["color_detected"] is True
    assert response["depth_detected"] is True
    assert response["color_corners"] == [[10.0, 10.0], [110.0, 12.0], [108.0, 90.0], [12.0, 88.0]]
    assert response["depth_corners"] == [[8.0, 14.0], [95.0, 11.0], [101.0, 80.0], [6.0, 83.0]]


def test_update_registration_action(monkeypatch, tmp_path):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=tmp_path / "homography.npy",
    )

    saved = {}

    def fake_save_registration(matrix, path):
        saved["matrix"] = matrix.copy()
        saved["path"] = path

    monkeypatch.setattr("qr_to_pos.server.save_registration", fake_save_registration)

    response = server.handle_json_message(
        {
            "action": "update_registration",
            "color_corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "depth_corners": [[2, 3], [12, 3], [12, 13], [2, 13]],
        }
    )

    assert response["action"] == "update_registration"
    assert response["saved_path"] == str(tmp_path / "homography.npy")
    assert len(response["homography"]) == 3
    np.testing.assert_allclose(saved["matrix"], np.array(response["homography"]))
    assert saved["path"] == str(tmp_path / "homography.npy")
