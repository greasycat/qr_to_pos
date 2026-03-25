import argparse
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request, send_file, url_for

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
REGISTRATION_DIR = ASSETS_DIR / "registration"
REGISTRATION_COORDS_PATH = REGISTRATION_DIR / "coords.yml"
CONFIG_PATH = REGISTRATION_DIR / "config.yml"
_VALID_DETECTOR_TYPES = {"qr", "apriltag"}
app = Flask(__name__)


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _normalize_corners(payload: object, field_name: str) -> list[list[float]]:
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError(f"Field '{field_name}' must be a list of 4 [x, y] points")

    normalized: list[list[float]] = []
    for point in payload:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Each point in '{field_name}' must be a [x, y] pair")
        x, y = point
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            raise ValueError(f"Each coordinate in '{field_name}' must be numeric")
        normalized.append([float(x), float(y)])
    return normalized


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_ws_url="ws://localhost:8765",
        test_image_url=url_for("test_image"),
        registration_sample_url=url_for("registration_sample"),
        registration_depth_text_url=url_for("registration_depth_text"),
        registration_coords_url=url_for("save_registration_coords"),
        detector_config_url=url_for("get_detector_config"),
    )


@app.route("/test-image")
def test_image():
    return send_file(ASSETS_DIR / "fake_background_multiple_qr.png")


@app.route("/registration-sample")
def registration_sample():
    return jsonify(
        {
            "color_image_url": url_for("registration_color_image"),
            "depth_text_url": url_for("registration_depth_text"),
        }
    )


@app.route("/registration-color-image")
def registration_color_image():
    return send_file(REGISTRATION_DIR / "1.png")


@app.route("/registration-depth-text")
def registration_depth_text():
    return send_file(REGISTRATION_DIR / "1.txt", mimetype="text/plain")


@app.route("/registration-coords", methods=["POST"])
def save_registration_coords():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object body"}), 400

    try:
        color_corners = _normalize_corners(payload.get("color_corners"), "color_corners")
        depth_corners = _normalize_corners(payload.get("depth_corners"), "depth_corners")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    REGISTRATION_COORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRATION_COORDS_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "color_corners": color_corners,
                "depth_corners": depth_corners,
            },
            handle,
            sort_keys=False,
        )

    return jsonify(
        {
            "saved_path": str(REGISTRATION_COORDS_PATH),
            "color_corners": color_corners,
            "depth_corners": depth_corners,
        }
    )


@app.route("/detector-config", methods=["GET"])
def get_detector_config():
    config = _read_config()
    detector_type = config.get("detector", {}).get("type", "qr")
    return jsonify({"type": detector_type})


@app.route("/detector-config", methods=["POST"])
def set_detector_config():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object body"}), 400

    detector_type = payload.get("type")
    if detector_type not in _VALID_DETECTOR_TYPES:
        return jsonify({"error": f"Invalid type. Must be one of: {sorted(_VALID_DETECTOR_TYPES)}"}), 400

    config = _read_config()
    config.setdefault("detector", {})["type"] = detector_type
    _write_config(config)
    return jsonify({"type": detector_type})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web frontend for QR detection WebSocket testing")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
