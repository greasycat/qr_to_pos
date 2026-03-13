# qr_to_pos Launch and Server Notes

This file documents the current startup flow implemented in `launch.py`, the QR WebSocket server in `qr_to_pos/server.py`, and the Flask backend in `web/app.py`.

## Launcher

Start the interactive launcher with:

```bash
uv run python launch.py
```

`launch.py` does not accept CLI arguments. It always starts these two services:

1. `QR WebSocket Server`
   Command:
   ```bash
   uv run python -m qr_to_pos.server
   ```
   Default endpoint: `ws://localhost:8765`

2. `Web UI (Flask)`
   Command:
   ```bash
   uv run python web/app.py
   ```
   Default endpoint: `http://localhost:5000`

Launcher key bindings:

- `j` or Down: move selection down
- `k` or Up: move selection up
- `s`: start selected service
- `x`: stop selected service
- `r`: restart selected service
- `o`: open the Flask UI in a browser
- `q` or `Ctrl+C`: quit and stop all services

## WebSocket Server

Start directly with:

```bash
uv run python -m qr_to_pos.server [--host HOST] [--port PORT] [--model-size {n,s,m,l}] [--save-decoding-images]
```

Implemented CLI options:

| Option | Default | Notes |
| --- | --- | --- |
| `--host` | `localhost` | Bind address for the WebSocket server |
| `--port` | `8765` | Listening port |
| `--model-size` | `s` | qrdet YOLO model size: `n`, `s`, `m`, or `l` |
| `--save-decoding-images` | `off` | Save each detect request image plus response metadata to the temp debug folder |

Internal defaults that are not exposed as CLI flags:

- `max_size = 16 * 1024 * 1024` bytes for incoming WebSocket messages
- `registration_path = assets/registration/homography.npy`
- `debug_capture_dir = $TMPDIR/qr_to_pos/ws_debug`

Supported WebSocket request styles:

1. Binary message
   Send raw image bytes such as PNG or JPEG. The server treats this as a detection request.

2. JSON message
   Send a JSON object with one of these actions:

- `detect`
  Required field: `image` (base64-encoded image bytes)
  Optional field: `flip_horizontal` (`false` by default)
- `update_corners`
  Required fields: `color_image` (base64 image), `depth_text` (tab-separated depth text)
- `update_registration`
  Required fields: `color_corners`, `depth_corners`
  Optional field: `save` (`true` by default)

Response behavior:

- `detect` returns `action`, `homography`, `detections`, `count`, and `processing_time`
- Each detection may include `bbox`, `confidence`, `decoded`, `homography`, `depth_bbox`, `depth_centroid`, and `depth_centroid_pct`
- Invalid JSON or missing fields return JSON errors

Debug capture settings live in [`assets/registration/config.yml`](/home/rongfei/WorkSpace/qr_to_pos/assets/registration/config.yml):

```yaml
ws_debug:
  save_decoding_images: false
  max_saved_images: 200
```

When enabled, each detect request is saved under the temp debug folder as a timestamped directory containing:

- `input.png`
- `response.json`
- `metadata.yml`

`metadata.yml` records whether `flip_horizontal` was applied before detection.

## Flask Backend

Start directly with:

```bash
uv run python web/app.py [--host HOST] [--port PORT]
```

Implemented CLI options:

| Option | Default | Notes |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address for the Flask app |
| `--port` | `5000` | Listening port |

Flask routes exposed by `web/app.py`:

- `/`
  Serves the test UI and injects `ws://localhost:8765` as the default WebSocket URL
- `/test-image`
  Returns `assets/fake_background_multiple_qr.png`
- `/registration-sample`
  Returns JSON with URLs for registration sample assets
- `/registration-color-image`
  Returns `assets/registration/1.png`
- `/registration-depth-text`
  Returns `assets/registration/1.txt`

## Practical Notes

- If you use `launch.py`, the launcher starts both servers with their defaults only.
- If you need custom host or port values, start `qr_to_pos.server` and `web/app.py` directly instead of using `launch.py`.
- The Flask UI assumes the QR WebSocket server is reachable at `ws://localhost:8765` unless changed in the browser UI.
