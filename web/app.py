from flask import Flask, jsonify, render_template, send_file, url_for
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
REGISTRATION_DIR = ASSETS_DIR / "registration"
app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_ws_url="ws://localhost:8765",
        test_image_url=url_for("test_image"),
        registration_sample_url=url_for("registration_sample"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web frontend for QR detection WebSocket testing")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
