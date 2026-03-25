# AGENTS.md

- Project name: `qr-to-pos`
- Primary language/runtime: Python 3.12, managed with `uv`
- Secondary: C# (Unity integration)

## Project Overview

- **QR WebSocket Server**: Detects and decodes QR codes (YOLO-based via `qrdet`), integrates with RealSense depth camera for 3D coordinates; listens on `ws://localhost:8765`
- **Flask Web UI**: Calibration/registration tool for tuning QR detection parameters; serves on `http://localhost:5000`
- **Unity Integration Client**: Remote machine with camera hardware runs Unity C# scripts; local machine can only run non-hardware or mocked tests

## Working Rules

- Use existing libraries (`opencv`, `pyzbar`, `qrdet`, `websockets`, `flask`, etc.) rather than reimplementing functionality.

### Version Control

- Always commit after completing a change, even if local tests were not run (some tests require remote hardware).
- Always use conventional commit messages: `feat(x): details`, `fix(x): details`, `docs:`, `chore:`, etc.
- Always create a new branch before starting a large change.
- Always append a one-liner to `CHANGELOG.md` for dev-friendly inspection (more verbose than the commit message).

### Editing

- Prefer small, targeted changes.
- Keep new code consistent with existing project structure and style.

### Communication

- Be concise and action-oriented.
- Summarize what changed and how it was verified.

## Important Commands

| Purpose | Command |
|---------|---------|
| Launch both services (TUI) | `uv run python launch.py` or `./run.sh` |
| QR WebSocket Server only | `uv run python -m qr_to_pos.server` |
| Flask Web UI only | `uv run python web/app.py` |
| Run tests | `uv run pytest` |

## Task-Specific Notes

- **Constraints**: Hardware-dependent tests (RealSense camera, Unity client) can only run on the remote machine; commit even without running them locally.
- **Preferences**: Use `uv` for all Python execution and dependency management.
- **Definition of done**: Code committed with conventional message, `CHANGELOG.md` updated, local non-hardware tests pass.
