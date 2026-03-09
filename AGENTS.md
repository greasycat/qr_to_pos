# AGENTS.md

# Python
use `uv` 

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the GUI (tkinter app that launches the WebSocket server)
./run.sh
# or
uv run python gui.py

# Run the interactive CLI (requires physical RealSense camera)
uv run python main.py

# Run the WebSocket server directly
uv run python -m qr_to_pos.server [--host HOST] [--port PORT] [--model-size {n,s,m,l}]

# Run tests
uv run pytest

# Run a single test file
uv run pytest path/to/test_file.py

# Install dependencies
uv sync
uv sync --group dev  # include dev dependencies
```

## Architecture

The project has two operational modes sharing the same `qr_to_pos` package:

**Mode Debugging –  RealSense camera pipeline** (`main.py`):
- `Camera` (`qr_to_pos/camera.py`) captures color frames from an Intel RealSense depth camera using `pyrealsense2`. It runs a background thread (`_capture_loop`) and exposes the latest frame via `get_latest_frame()` plus an `on_frame()` callback mechanism.
- `QRCodeProcessor` (`qr_to_pos/processor.py`) runs its own background thread polling `Camera.get_latest_frame()`, calls `qrdet.QRDetector` (YOLO-based) on each new frame, and emits `ProcessingResult` objects via `on_result()` callbacks. Rate-limited by `min_interval`.
- `main.py` wires these together with an interactive CLI and a side-by-side raw/annotated OpenCV visualization window.

**Mode 2 – Actual WebSocket server** (`qr_to_pos/server.py`):
- `DetectionServer` is a standalone async WebSocket server (using `websockets`). Clients send images as raw bytes or as base64-encoded JSON `{"image": "..."}`.
- Detection: `qrdet.QRDetector` locates QR bounding boxes → `pyzbar` decodes QR content from a cropped/padded grayscale region. Returns JSON with `detections`, `count`, and `processing_time`.
- `gui.py` is a Tkinter frontend that launches the server as a subprocess (`python -m qr_to_pos.server`).

**Package exports** (`qr_to_pos/__init__.py`): `Camera`, `Frame`, `QRCodeProcessor`, `ProcessingResult`, `QRCode`, `DetectionServer`.

## Key Details

- Python 3.12, managed with `uv`.
- The `stubs/pyrealsense2/` directory contains hand-written type stubs for `pyrealsense2` (no upstream stubs). `pyproject.toml` configures `[tool.ty.environment] extra-paths = ["./stubs"]`.
- `qr_to_pos/qr.py` is a standalone test script for static image QR detection (not part of the importable package).
- `QRCode.data` comes from `qrdet` raw detection output; `QRCode.decoded` (server mode only) is the human-readable string decoded by `pyzbar`.
- The `Camera` class requires a RealSense device with a Color sensor at init time — it validates this before starting. Tests or non-camera usage must mock this.
