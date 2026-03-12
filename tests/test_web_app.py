from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.app import app


def test_save_registration_coords_writes_yaml(monkeypatch, tmp_path):
    coords_path = tmp_path / "coords.yml"
    monkeypatch.setattr("web.app.REGISTRATION_COORDS_PATH", coords_path)

    client = app.test_client()
    response = client.post(
        "/registration-coords",
        json={
            "color_corners": [[1, 2], [3.5, 4], [5, 6], [7, 8]],
            "depth_corners": [[11, 12], [13, 14], [15, 16], [17, 18]],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["saved_path"] == str(coords_path)

    saved = yaml.safe_load(coords_path.read_text(encoding="utf-8"))
    assert saved == {
        "color_corners": [[1.0, 2.0], [3.5, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "depth_corners": [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
    }


def test_save_registration_coords_rejects_invalid_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("web.app.REGISTRATION_COORDS_PATH", tmp_path / "coords.yml")

    client = app.test_client()
    response = client.post(
        "/registration-coords",
        json={
            "color_corners": [[1, 2], [3, 4], [5, 6]],
            "depth_corners": [[11, 12], [13, 14], [15, 16], [17, 18]],
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Field 'color_corners' must be a list of 4 [x, y] points"
    }
