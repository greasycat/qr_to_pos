2026-03-25: Use the normal `detect` websocket payload for Unity debug-image requests so manually loaded textures bypass Unity flip processing.
2026-03-25: Normalize QR and AprilTag bounding boxes from decoded quad geometry so both detector modes project detections consistently.
2026-03-25: Add a synthetic QR-vs-AprilTag bbox regression test that holds detector placement deltas within a fixed epsilon.
