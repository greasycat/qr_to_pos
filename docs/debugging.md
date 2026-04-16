# Debugging

## Python Side

### Server-side debug captures

Enable frame capture to inspect exactly what the server receives and detects:

```bash
uv run python -m marker_to_pos.server --save-decoding-images
```

Or set it in `assets/registration/config.yml`:

```yaml
ws_debug:
  save_decoding_images: true
  max_saved_images: 200
```

Each `detect` or `detect_unity` request saves a timestamped directory under `$TMPDIR/marker_to_pos/ws_debug/` containing:

| File | Contents |
|---|---|
| `input.png` | The preprocessed image the detector actually sees |
| `processed.png` | Same image with detected bounding boxes drawn as ellipses |
| `response.json` | Full detection response JSON |
| `metadata.yml` | Timestamp, request source, image shape, detection count, processing time |

The oldest directories are pruned automatically when `max_saved_images` is exceeded.

To find the debug directory:

```python
import tempfile, pathlib
print(pathlib.Path(tempfile.gettempdir()) / "marker_to_pos" / "ws_debug")
```

### Tests

Run the test suite without any hardware:

```bash
uv run python -m pytest
```

Key test files:
- `tests/test_server.py` — WebSocket server request/response handling
- `tests/test_registration.py` — homography and corner detection
- `tests/test_web_app.py` — Flask route behavior

All tests use fixtures or sample assets; no camera connection is required.

### Inspecting detection geometry

`marker_to_pos/inspection.py` contains utilities for visualizing detection bounding boxes and transformed depth coordinates against registration assets. Run directly or import in a notebook for exploratory work.

---

## Unity Side

### Debug Mode

`MarkerDetectionRenderer` has a **Debug Mode** field with three options:

| Mode | Use |
|---|---|
| `None` | Production — uses live RealSense source |
| `MockServer` | Sends dummy frames; relies on an external WebSocket server (e.g. `mock_backend.py`) for detections |
| `SingleImage` | Sends a single static image repeatedly; good for checking detection without live feed |

Use **MockServer** with `uv run python mock_backend.py` to iterate on placement, construction bindings, and terrain mapping without hardware.

### Debug Bounds

Enable **Show Debug Bounds** on `MarkerDetectionRenderer` to spawn 4 colored cubes at the terrain corners of the detection region. These mark where the 0% and 100% normalized coordinates land in world space, making it easy to verify that the coordinate mapping and flip settings are correct.

Configure the bounds display with:
- **Debug Bounds Color** — color of the corner cubes
- **Debug Bounds Scale** — size of each corner cube
- **Debug Bounds Y** — fixed Y height of the corner cubes (independent of terrain)

### Inspector diagnostics

`MarkerDetectionRenderer` exposes two read-only Inspector fields updated each frame:

- **Detection Count** — number of tags returned in the last WebSocket response
- **Out Of Bounds Conversion Count** — number of detections whose `depth_centroid_pct` fell outside [0, 100] and were skipped

If Detection Count stays at 0, check the WebSocket connection and server logs. If Out Of Bounds is consistently non-zero, the registration homography may need recalibration.

### Manual image loader (Editor window)

The Editor script `unity/Editor/MarkerManualImageLoaderWindow.cs` provides a Unity Editor window for loading and sending test images to the detection server without entering Play mode. Open it from the Unity menu after importing the scripts.

### Marker lifetime

Markers are removed after **3 seconds** without a new detection (`MarkerManager.MarkerLifetimeSeconds`). If markers are disappearing unexpectedly:
- Check that the server is receiving and responding to frames (look at server stdout)
- Verify **Send Interval** on `MarkerDetectionRenderer` is short enough relative to the 3-second lifetime

### Wall construction not appearing

- Confirm both marker indexes assigned to the wall binding are actively detected in the same frame
- Check that the **Terrain** field is assigned — wall grounding uses `terrain.SampleHeight()` and silently skips if terrain is null
- If the wall jumps each frame, verify that `flipX`/`flipZ` are stable and that the two markers have consistent `depth_centroid_pct` values (watch server stdout or enable debug captures)
