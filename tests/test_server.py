import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marker_to_pos.server import DetectionServer

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


class _DummyQuadDetector:
    def detect(self, image, is_bgr=True):
        return [
            {
                "bbox_xyxy": [10, 20, 90, 80],
                "quad_xy": [[12, 24], [88, 22], [87, 78], [11, 79]],
                "confidence": 0.97,
                "data": "dummy",
            }
        ]


class _DummyDecode:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _SpyDetector:
    def __init__(self) -> None:
        self.images = []

    def detect(self, image, is_bgr=True):
        self.images.append(image.copy())
        return []


def _write_server_config(
    path: Path,
    *,
    save_decoding_images: bool = False,
    max_saved_images: int = 2,
    unity_image_actions: list[str] | None = None,
    detector_type: str | None = None,
) -> None:
    payload = {
        "ws_debug": {
            "save_decoding_images": save_decoding_images,
            "max_saved_images": max_saved_images,
        }
    }
    if unity_image_actions is not None:
        payload["ws_processing"] = {
            "unity_image_actions": unity_image_actions,
        }
    if detector_type is not None:
        payload["detector"] = {"type": detector_type}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture()
def stubbed_server(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])
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
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

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


def test_detect_prefers_quad_extents_for_bbox(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyQuadDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=tmp_path / "homography.npy",
    )

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    qr_codes = server.detect(image)

    assert len(qr_codes) == 1
    assert qr_codes[0].bbox == (11, 22, 88, 79)


def test_detect_request_saves_debug_capture_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    capture_dir = tmp_path / "captures"
    _write_server_config(
        config_path,
        save_decoding_images=True,
        max_saved_images=2,
    )

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=registration_path,
        debug_capture_dir=capture_dir,
    )

    image = server.decode_image(IMAGE_PATH.read_bytes())
    response = server.process_detect_request(image, request_source="json")

    capture_dirs = sorted(path for path in capture_dir.iterdir() if path.is_dir())
    assert len(capture_dirs) == 1

    saved_capture_dir = capture_dirs[0]
    assert (saved_capture_dir / "input.png").exists()
    assert (saved_capture_dir / "processed.png").exists()
    assert (saved_capture_dir / "response.json").exists()
    assert (saved_capture_dir / "metadata.yml").exists()

    saved_response = json.loads((saved_capture_dir / "response.json").read_text(encoding="utf-8"))
    assert saved_response["count"] == response["count"]
    assert saved_response["detections"][0]["decoded"] == "stubbed-qr"

    metadata = yaml.safe_load((saved_capture_dir / "metadata.yml").read_text(encoding="utf-8"))
    assert metadata["request_source"] == "json"
    assert metadata["request_action"] == "detect"
    assert metadata["image_shape"] == list(image.shape)
    assert metadata["count"] == response["count"]

    saved_input = cv2.imread(str(saved_capture_dir / "input.png"), cv2.IMREAD_COLOR)
    saved_processed = cv2.imread(str(saved_capture_dir / "processed.png"), cv2.IMREAD_COLOR)
    assert saved_input is not None
    assert saved_processed is not None
    assert np.any(saved_processed != saved_input)


def test_detect_request_prunes_debug_captures_to_configured_limit(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    capture_dir = tmp_path / "captures"
    _write_server_config(
        config_path,
        save_decoding_images=True,
        max_saved_images=2,
    )

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=registration_path,
        debug_capture_dir=capture_dir,
    )

    image = server.decode_image(IMAGE_PATH.read_bytes())
    server.process_detect_request(image, request_source="json-1")
    server.process_detect_request(image, request_source="json-2")
    server.process_detect_request(image, request_source="json-3")

    capture_dirs = sorted(path for path in capture_dir.iterdir() if path.is_dir())
    assert len(capture_dirs) == 2

    saved_sources = [
        yaml.safe_load((path / "metadata.yml").read_text(encoding="utf-8"))["request_source"]
        for path in capture_dirs
    ]
    assert saved_sources == ["json-2", "json-3"]


def test_compute_depth_centroid_pct_returns_none_outside_registered_bounds(monkeypatch):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
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


def test_detect_action_does_not_flip_image(monkeypatch, tmp_path):
    detector = _SpyDetector()
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": detector)
    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=tmp_path / "homography.npy",
    )

    image = np.zeros((140, 140, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = server.handle_json_message(
        {
            "action": "detect",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )

    assert response["action"] == "detect"
    assert len(detector.images) == 1
    np.testing.assert_array_equal(detector.images[0], image)


def test_detect_unity_action_flips_image_horizontally(monkeypatch, tmp_path):
    detector = _SpyDetector()
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": detector)
    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=tmp_path / "homography.npy",
    )

    image = np.zeros((140, 140, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = server.handle_json_message(
        {
            "action": "detect_unity",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )

    assert response["action"] == "detect_unity"
    assert len(detector.images) == 1
    np.testing.assert_array_equal(detector.images[0], cv2.flip(image, 1))


def test_detect_unity_action_uses_yaml_image_action_pipeline(monkeypatch, tmp_path):
    detector = _SpyDetector()
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": detector)

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    _write_server_config(
        config_path,
        unity_image_actions=["flip_h", "flip_v"],
    )

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=registration_path,
    )

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    image[0, 4] = [40, 50, 60]
    image[3, 0] = [70, 80, 90]
    image[3, 4] = [100, 110, 120]
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = server.handle_json_message(
        {
            "action": "detect_unity",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )

    assert response["action"] == "detect_unity"
    assert len(detector.images) == 1
    expected = cv2.flip(cv2.flip(image, 1), 0)
    np.testing.assert_array_equal(detector.images[0], expected)


def test_invalid_unity_image_action_in_yaml_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _SpyDetector())

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    _write_server_config(
        config_path,
        unity_image_actions=["flip_h", "nope"],
    )

    with pytest.raises(ValueError, match="Unsupported unity image action: nope"):
        DetectionServer(
            host="localhost",
            port=0,
            model_size="s",
            registration_path=registration_path,
        )


def test_detect_unity_request_saves_unity_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [_DummyDecode(b"stubbed-qr")])

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    capture_dir = tmp_path / "captures"
    _write_server_config(
        config_path,
        save_decoding_images=True,
        max_saved_images=2,
    )

    server = DetectionServer(
        host="localhost",
        port=0,
        model_size="s",
        registration_path=registration_path,
        debug_capture_dir=capture_dir,
    )

    image = np.zeros((140, 140, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = server.handle_json_message(
        {
            "action": "detect_unity",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )

    assert response["action"] == "detect_unity"
    capture_dirs = sorted(path for path in capture_dir.iterdir() if path.is_dir())
    assert len(capture_dirs) == 1

    saved_capture_dir = capture_dirs[0]
    metadata = yaml.safe_load((saved_capture_dir / "metadata.yml").read_text(encoding="utf-8"))
    assert metadata["request_source"] == "unity"
    assert metadata["request_action"] == "detect_unity"

    saved_image = cv2.imread(str(saved_capture_dir / "input.png"), cv2.IMREAD_COLOR)
    saved_processed = cv2.imread(str(saved_capture_dir / "processed.png"), cv2.IMREAD_COLOR)
    assert saved_image is not None
    assert saved_processed is not None
    np.testing.assert_array_equal(saved_image, cv2.flip(image, 1))
    assert np.any(saved_processed != saved_image)


def test_update_corners_action(monkeypatch):
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
    server = DetectionServer(host="localhost", port=0, model_size="s")

    monkeypatch.setattr(
        "marker_to_pos.server.detect_box_corners_color",
        lambda _image: np.array([[10, 10], [110, 12], [108, 90], [12, 88]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "marker_to_pos.server.detect_box_corners_depth",
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
    monkeypatch.setattr("marker_to_pos.server.QRDetector", lambda model_size="s": _DummyDetector())
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

    monkeypatch.setattr("marker_to_pos.server.save_registration", fake_save_registration)

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


# ── AprilTag detector tests ──────────────────────────────────────────────────


class _DummyAprilTagDetector:
    """Minimal AprilTagDetector stand-in for tests."""

    def detect(self, image, *, is_bgr=True):
        return [
            {
                "bbox_xyxy": (10.0, 20.0, 90.0, 80.0),
                "confidence": 42.5,
                "data": "7",
                "_decoded": "tag36h11:7",
            }
        ]


def test_config_selects_apriltag_detector(monkeypatch, tmp_path):
    sentinel = _DummyAprilTagDetector()
    monkeypatch.setattr(
        "marker_to_pos.apriltag_detector.AprilTagDetector",
        lambda **_kwargs: sentinel,
    )

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    _write_server_config(config_path, detector_type="apriltag")

    server = DetectionServer(host="localhost", port=0, registration_path=registration_path)

    assert server.detector is sentinel


def test_apriltag_result_skips_pyzbar(monkeypatch, tmp_path):
    """When a detection dict contains _decoded, pyzbar must not be called."""
    monkeypatch.setattr(
        "marker_to_pos.apriltag_detector.AprilTagDetector",
        lambda **_kwargs: _DummyAprilTagDetector(),
    )

    pyzbar_called = []
    monkeypatch.setattr(
        "marker_to_pos.server.pyzbar_decode",
        lambda _crop: pyzbar_called.append(True) or [],
    )

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    _write_server_config(config_path, detector_type="apriltag")

    server = DetectionServer(host="localhost", port=0, registration_path=registration_path)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    qr_codes = server.detect(image)

    assert not pyzbar_called, "pyzbar_decode should not be called for AprilTag detections"
    assert len(qr_codes) == 1


def test_apriltag_result_mapping(monkeypatch, tmp_path):
    """AprilTag detection is correctly mapped to QRCode fields."""
    monkeypatch.setattr(
        "marker_to_pos.apriltag_detector.AprilTagDetector",
        lambda **_kwargs: _DummyAprilTagDetector(),
    )
    monkeypatch.setattr("marker_to_pos.server.pyzbar_decode", lambda _crop: [])

    registration_path = tmp_path / "homography.npy"
    config_path = tmp_path / "config.yml"
    _write_server_config(config_path, detector_type="apriltag")

    server = DetectionServer(host="localhost", port=0, registration_path=registration_path)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    qr_codes = server.detect(image)

    assert len(qr_codes) == 1
    qr = qr_codes[0]
    assert qr.data == "7"
    assert qr.decoded == "tag36h11:7"
    assert qr.bbox == (10, 20, 90, 80)
    assert qr.confidence == 42.5
