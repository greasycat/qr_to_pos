# qr_to_pos

Real-time QR code detection from Intel RealSense cameras, with a WebSocket server for remote image processing.

## Setup

Requires Python 3.12+.

```bash
uv sync
```

## WebSocket Detection Server

Accepts images over WebSocket and returns QR code detection results.

### Start the server

```bash
python -m qr_to_pos.server
```

Options:

| Flag             | Default     | Description                          |
| ---------------- | ----------- | ------------------------------------ |
| `--host`         | `localhost` | Bind address                         |
| `--port`         | `8765`      | Port number                          |
| `--model-size`   | `s`         | YOLO model size (`n`, `s`, `m`, `l`) |

### Send images

The server accepts two input formats:

**Binary** — send raw image bytes (PNG, JPEG, etc.) as a binary WebSocket message.

**JSON** — send a text message with a base64-encoded image:

```json
{"image": "<base64-encoded image bytes>"}
```

### Response

```json
{
  "detections": [
    {
      "data": "https://example.com",
      "bbox": [100, 200, 300, 400],
      "confidence": 0.95,
      "decoded": "https://example.com"
    }
  ],
  "count": 1,
  "processing_time": 0.0342
}
```

- `data` — raw value from the YOLO detection model (may be empty)
- `decoded` — QR content decoded by pyzbar (`null` if decoding failed)

On error: `{"error": "description"}`.

### Python client example

```python
import asyncio
import websockets

async def detect(image_path: str):
    async with websockets.connect("ws://localhost:8765") as ws:
        with open(image_path, "rb") as f:
            await ws.send(f.read())
        print(await ws.recv())

asyncio.run(detect("qrs.png"))
```
