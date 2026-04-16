# Server Payloads

WebSocket protocol reference for the AprilTag detection server (`marker_to_pos.server`).

## Connection

Default endpoint: `ws://localhost:8765`

All messages are either raw binary (legacy) or UTF-8 JSON strings.

## Requests

### Binary message (legacy)

Send raw image bytes (PNG or JPEG). The server treats this as a `detect` request with no image preprocessing.

### JSON messages

Send a UTF-8 JSON object with an `action` field.

---

#### `detect`

Run AprilTag detection on a single image. No preprocessing is applied (used by the Flask calibration UI).

```json
{
  "action": "detect",
  "image": "<base64-encoded PNG or JPEG>"
}
```

If `action` is omitted but `image` is present, the server defaults to `detect`.

---

#### `detect_unity`

Run AprilTag detection with the configured `unity_image_actions` preprocessing pipeline applied first (default: horizontal flip). Used by the Unity live renderer.

```json
{
  "action": "detect_unity",
  "image": "<base64-encoded PNG or JPEG>"
}
```

---

#### `update_corners`

Detect the 4 sandbox corners in a color+depth frame pair. Used by the calibration UI.

```json
{
  "action": "update_corners",
  "color_image": "<base64-encoded PNG or JPEG>",
  "depth_text": "<tab-separated depth values as plain text>"
}
```

Depth text format: one header line (ignored), then rows of tab-separated float values.

---

#### `update_registration`

Compute and save a new homography from manually provided corner points.

```json
{
  "action": "update_registration",
  "color_corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "depth_corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "save": true
}
```

`save` defaults to `true`. Set to `false` to compute the homography without writing it to disk.

---

## Responses

### `detect` / `detect_unity` response

```json
{
  "action": "detect",
  "homography": [[...], [...], [...]],
  "count": 2,
  "processing_time": 0.0043,
  "detections": [
    {
      "data": "42",
      "bbox": [120, 80, 200, 160],
      "confidence": 0.99,
      "decoded": "tag36h11:42",
      "homography": [[...], [...], [...]],
      "depth_bbox": [[x,y], [x,y], [x,y], [x,y]],
      "depth_centroid": [160.0, 120.0],
      "depth_centroid_pct": [25.0, 18.75]
    }
  ]
}
```

Detection fields:

| Field | Type | Description |
|---|---|---|
| `data` | string | Tag ID (e.g. `"42"`) — used as the tracking key in Unity |
| `bbox` | `[x1, y1, x2, y2]` | Bounding box in the color frame (pixels) |
| `confidence` | float | Detector confidence score |
| `decoded` | string | Human-readable label (e.g. `"tag36h11:42"`) |
| `homography` | 3×3 float array | Color-to-depth homography matrix |
| `depth_bbox` | 4×2 float array | Bounding quad projected into depth frame |
| `depth_centroid` | `[x, y]` | Centroid in depth frame pixels |
| `depth_centroid_pct` | `[x_pct, y_pct]` | Centroid as percentage of depth frame (0–100 each axis) — used by Unity for terrain placement |

`homography`, `depth_bbox`, `depth_centroid`, and `depth_centroid_pct` are `null` when no registration homography is loaded.

---

### `update_corners` response

```json
{
  "action": "update_corners",
  "color_corners": [[x,y], [x,y], [x,y], [x,y]],
  "depth_corners": [[x,y], [x,y], [x,y], [x,y]],
  "color_detected": true,
  "depth_detected": false
}
```

`color_corners` or `depth_corners` is `null` when auto-detection failed for that frame.

---

### `update_registration` response

```json
{
  "action": "update_registration",
  "homography": [[...], [...], [...]],
  "saved_path": "assets/registration/homography.npy"
}
```

`saved_path` is `null` when `save` was `false`.

---

### Error response

Any request that fails returns:

```json
{
  "error": "description of the error",
  "action": "detect"
}
```

## Mock Backend Payload Format

`mock_backend.py` emits the same response schema as the real server but with synthetic detections and `homography: null`. The `depth_centroid_pct` values drive terrain placement directly:

```python
{
  "data": "3",
  "bbox": [...],
  "confidence": 0.99,
  "decoded": "tag36h11:3",
  "depth_centroid": [float, float],
  "depth_centroid_pct": [x_pct, y_pct]
}
```

Marker positions in the mock:

| Tag | Pattern |
|---|---|
| `1` | Fixed at ~(30, 30) with Gaussian jitter (σ=1.5) |
| `2` | X sweeps 10→90 sinusoidally, Y fixed at 50 |
| `3` | Orbits centre (65, 65) at radius 20 |
| `4` | Same orbit as tag 3 but 180° out of phase |
