# marker-to-pos

Detects AprilTags from a RealSense camera, streams detections over WebSocket, and drives live marker visualization in Unity.

<img width="684" height="350" alt="image" src="https://github.com/user-attachments/assets/55daba90-4e72-4460-b0e0-f6d45aa96e7d" />

## Python Backend

Install dependencies and start both services with:

```bash
uv sync
uv run python launch.py
```

The launcher starts:
- **WebSocket server** at `ws://localhost:8765` — receives images from Unity, runs AprilTag detection, returns detections
- **Flask UI** at `http://localhost:5000` — calibration and registration workflows

To run services separately:

```bash
uv run python -m marker_to_pos.server   # WebSocket only
uv run python web/app.py                # Flask UI only
```

For development without a camera, use the mock backend instead of `launch.py`:

```bash
uv run python mock_backend.py
```

This emits 4 synthetic markers (jitter, sweep, and two orbiting) at `ws://localhost:8765`.

## Calibration

Calibration maps the color camera frame to the depth frame using a 4-corner homography. Run once and the result persists to `assets/registration/homography.npy`.

**Option A — Flask UI (recommended):**
1. Open `http://localhost:5000`
2. Point the camera at the registration surface with visible corner markers
3. Use the registration workflow to detect or manually pick the 4 corners
4. Save registration

**Option B — CLI:**

```bash
uv run python scripts/register.py \
    --color assets/registration/1.png \
    --depth assets/registration/1.txt \
    --out   assets/registration/homography.npy
```

Add `--manual` to pick corners interactively instead of auto-detecting.

See [docs/calibration.md](docs/calibration.md) for tuning tips, coordinate remapping, and the full set of calibration scripts.

## Unity Renderer Quick Setup

The Unity scripts live in `unity/` (symlinked from the Sandbox Marker Unity project under `Assets/unity/`).

### 1. Add the component

Add `MarkerDetectionRenderer` to any GameObject in your scene.

### 2. Select a source

Set the **Debug Mode** field to choose how the renderer receives images:

| Debug Mode | Source |
|---|---|
| `None` | Live RealSense camera via `RsFrameProvider` — assign the `Source` field |
| `MockServer` | Sends dummy frames to the WebSocket server; works with `mock_backend.py` |
| `SingleImage` | Sends a single static image; useful for inspecting detection without live data |

For live use, set **Debug Mode** to `None` and drag the active `RsFrameProvider` component into the **Source** field.

### 3. Set server URL

Set **Server Url** to match your backend, e.g. `ws://localhost:8765`.

### 4. Assign terrain

Drag the active `Terrain` object into the **Terrain** field. Marker positions are sampled from the terrain heightmap at runtime.

### 5. Set up marker construction bindings

**Marker Construction Bindings** controls what appears in the scene for each detected tag.

Each binding has:
- **Name** — label used for GameObject naming
- **Marker Indexes** — which tag IDs this binding applies to (e.g. `[1]` for tag `1`)
- **Choice** — `Prefab` or `Wall`

**Prefab binding:** drag a prefab into the **Prefab** field. The prefab is instantiated at the marker's terrain position and respawned if it moves far enough.

**Wall binding:** list two or more marker indexes as wall endpoints. Configure the **Wall** sub-fields:
- `Height` — wall height in world units
- `Thickness` — wall thickness
- `Invert Y` — extend wall downward instead of upward (useful for ceilings or hanging elements)
- `Texture` — optional `Texture2D` applied to all wall faces

Leave **Marker Construction Bindings** empty to render all detected tags as plain colored cubes.

## Documentation

- [docs/calibration.md](docs/calibration.md) — calibration tuning, coordinate mapping, flipX/flipZ, vertical offset
- [docs/server-payloads.md](docs/server-payloads.md) — WebSocket protocol, all actions and response fields
- [docs/debugging.md](docs/debugging.md) — debug captures in Python, debug modes and bounds in Unity

## Repository Layout

```
marker_to_pos/   Python server, detection, registration, and processing
web/             Flask UI and templates
unity/           Unity runtime scripts, scene, terrain assets
  Scripts/       C# scripts
  Misc/          Terrain data, textures, materials
  Scene/         Unity scene files
scripts/         CLI tools for calibration and corner picking
tests/           Regression tests (no hardware required)
assets/          Sample images and registration files
mock_backend.py  Synthetic WebSocket server for Unity development
```
