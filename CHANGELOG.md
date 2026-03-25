2026-03-25: Rename all remaining QR-prefixed Unity files/classes and Python QRCode types to Marker-prefixed equivalents.
2026-03-25: Remove the redundant CLAUDE.md instructions file and keep AGENTS.md as the single repo guidance source.
2026-03-25: Rename the Unity renderer component to MarkerDetectionRenderer and strip inspector grouping attributes from the script.
2026-03-25: Remove QR detection support, make the stack AprilTag-only, and replace the bundled QR sample with an AprilTag test image.
2026-03-25: Use the normal `detect` websocket payload for Unity debug-image requests so manually loaded textures bypass Unity flip processing.
2026-03-25: Normalize QR and AprilTag bounding boxes from decoded quad geometry so both detector modes project detections consistently.
2026-03-25: Add a synthetic QR-vs-AprilTag bbox regression test that holds detector placement deltas within a fixed epsilon.
