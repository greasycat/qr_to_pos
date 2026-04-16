# Calibration

This document covers calibration workflows, coordinate system tuning, and runtime placement parameters.

## Overview

Calibration produces a homography matrix that maps 2D points in the color camera frame to 2D points in the depth frame. This is stored at `assets/registration/homography.npy` and loaded automatically by the WebSocket server at startup.

Without a valid homography, `depth_centroid_pct` in detection responses will be `null`, and Unity markers will not appear.

## Registration Scripts

### Flask UI (recommended for hardware setup)

Start the Flask UI and navigate to `http://localhost:5000`. The registration workflow guides you through:
1. Capturing a color and depth frame from the camera
2. Auto-detecting or manually picking the 4 sandbox corners (TL, TR, BR, BL order)
3. Computing and saving the homography

### CLI (`scripts/register.py`)

For scripted or offline registration:

```bash
uv run python scripts/register.py \
    --color assets/registration/1.png \
    --depth assets/registration/1.txt \
    --out   assets/registration/homography.npy
```

Flags:
- `--manual` — skip auto-detection, open interactive OpenCV windows to pick corners by clicking
- Without `--manual`, auto-detection runs first; if it fails on either frame, the script falls back to manual picking

### Color corner tuning (`scripts/tune_color_corners.py`)

Adjust the color corner detection parameters interactively. Run against a sample image when auto-detection is unreliable due to lighting or surface color variation.

### Depth corner tuning (`scripts/tune_depth_corners.py`)

Interactive tuner for the depth frame corner detector. Useful when the sandbox walls produce inconsistent depth readings.

### Corner picking utility (`scripts/pick_depth_corners.py`)

Standalone picker for the depth frame only. Use when you want to re-register the depth corners without re-processing color.

## Coordinate Mapping

The WebSocket server returns `depth_centroid_pct` as `[x_pct, y_pct]` in the depth frame (0–100 each axis). Unity maps these to terrain world coordinates via `MarkerTerrainMapper`:

```
xNorm = centroidYPct / 100    (depth Y → terrain X)
zNorm = centroidXPct / 100    (depth X → terrain Z)
```

The axes are swapped by design: the depth image's horizontal axis corresponds to the terrain's Z axis, and vice versa. This matches the physical orientation of the sandbox relative to the overhead camera in the reference setup.

### flipX and flipZ

If the detected marker positions appear mirrored along one axis, adjust the **Flip X** or **Flip Z** checkboxes on `MarkerDetectionRenderer`:

- `flipX = true` — inverts xNorm: `xNorm = 1 - xNorm`
- `flipZ = true` — inverts zNorm: `zNorm = 1 - zNorm`

Both default to `false`. Enable one or both if your camera or sandbox is mounted differently from the reference orientation.

### Marker Vertical Offset

**Marker Vertical Offset** (default `0.05`) is added to the terrain surface Y when placing markers. Set it slightly above zero to prevent z-fighting between the marker and the terrain surface.

For large prefabs or visual effects that should appear elevated, increase this value. The renderer snaps the marker Y to the terrain surface on every frame, so large offset values produce instant correction rather than gradual settling.

## config.yml

Server-side processing is configured in `assets/registration/config.yml`:

```yaml
ws_debug:
  save_decoding_images: false
  max_saved_images: 200

ws_processing:
  unity_image_actions:
    - flip_h
```

### unity_image_actions

Applied to every image received via the `detect_unity` action before running the AprilTag detector. Order matters — actions are applied left to right.

| Action | Effect |
|---|---|
| `flip_h` | Horizontal flip (left↔right) |
| `flip_v` | Vertical flip (top↔bottom) |
| `r_180_plus` | Rotate 90° clockwise |
| `r_180_minus` | Rotate 90° counterclockwise |

The default is `[flip_h]`. Adjust if your camera image arrives with a different orientation than expected.
