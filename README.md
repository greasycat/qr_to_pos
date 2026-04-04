# marker-to-pos

`marker-to-pos` detects AprilTags from camera images, exposes detections over WebSocket, provides a Flask calibration UI, and includes Unity-side scripts for consuming live marker updates.

## Components

- `marker_to_pos.server`: AprilTag WebSocket server, with RealSense-assisted registration and depth-aware detection data.
- `web/app.py`: Flask UI for calibration, detector configuration, and registration workflows.
- `unity/`: Unity scripts for live marker visualization and scene integration on the remote hardware machine.

## Quick Start

Use `uv` for all Python setup and execution:

```bash
uv sync
uv run python launch.py
```

Alternative entry points:

- Launch both services: `uv run python launch.py` or `./run.sh`
- WebSocket server only: `uv run python -m marker_to_pos.server`
- Flask UI only: `uv run python web/app.py`
- Run tests: `uv run python -m pytest`

Default local endpoints:

- WebSocket server: `ws://localhost:8765`
- Flask UI: `http://localhost:5000`

## Hardware Notes

- The full stack expects the camera hardware and Unity client to run on the remote machine.
- This local repo can still run non-hardware code paths, the Flask app, and mocked tests.

## Documentation

- Detailed launcher, WebSocket, and Flask runtime reference: [docs/runtime-reference.md](docs/runtime-reference.md)

## Repository Layout

- `marker_to_pos/`: Python server, processing, detection, and registration modules
- `web/`: Flask app and templates
- `unity/`: Unity runtime and editor scripts
- `tests/`: Local non-hardware regression coverage
- `assets/`: Sample images, registration assets, and configuration
