import base64
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

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
def stubbed_server(monkeypatch, tmp_path):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("qr_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])
    return DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=tmp_path / "homography.npy",
    )


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
    assert result["homography"] is None

    for det in result["detections"]:
        # Every detection must have a bounding box with 4 ints
        assert det["bbox"] is not None
        assert len(det["bbox"]) == 4
        assert all(isinstance(c, int) for c in det["bbox"])

        # Confidence should be a positive number
        assert det["confidence"] is not None
        assert det["confidence"] > 0
        assert det["homography"] is None
        assert det["depth_bbox"] is None
        assert det["depth_centroid"] is None
        assert det["depth_centroid_pct"] is None

        # pyzbar should have decoded at least some of them
    decoded_values = [d["decoded"] for d in result["detections"] if d["decoded"]]
    assert len(decoded_values) > 0, "pyzbar should decode at least one QR code"

    print(decoded_values)


def test_detect_qrs_includes_registration_projection(monkeypatch, tmp_path):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("qr_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

    registration_path = tmp_path / "homography.npy"
    coords_path = tmp_path / "coords.yml"
    homography = np.array(
        [
            [1.0, 0.0, 100.0],
            [0.0, 1.0, 200.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.save(registration_path, homography)
    coords_path.write_text(
        yaml.safe_dump(
            {
                "depth_corners": [
                    [100.0, 200.0],
                    [200.0, 200.0],
                    [200.0, 300.0],
                    [100.0, 300.0],
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=registration_path,
    )

    image_bytes = IMAGE_PATH.read_bytes()
    image = server.decode_image(image_bytes)
    result = server.detect_response(image)

    assert result["homography"] == homography.tolist()
    assert len(result["detections"]) == 1
    detection = result["detections"][0]
    assert detection["homography"] == homography.tolist()
    assert detection["depth_bbox"] == [
        [112.0, 218.0],
        [196.0, 218.0],
        [196.0, 304.0],
        [112.0, 304.0],
    ]
    assert detection["depth_centroid"] == [154.0, 261.0]
    assert detection["depth_centroid_pct"] == [54.0, 61.0]


def test_compute_depth_centroid_pct_returns_none_outside_registered_bounds(monkeypatch):
    monkeypatch.setattr("qr_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    server = DetectionServer(host="localhost", port=0, model_size="s")

    depth_bbox = np.array(
        [
            [150.0, 120.0],
            [170.0, 120.0],
            [170.0, 140.0],
            [150.0, 140.0],
        ],
        dtype=np.float32,
    )
    depth_corners = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
        dtype=np.float32,
    )

    assert server._compute_depth_centroid_pct(depth_bbox, depth_corners) is None


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
