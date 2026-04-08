2026-04-08: Update Unity procedural walls to use a shared unit prism mesh with transform-based position, rotation, and scale updates instead of rebuilding wall vertices every refresh.
2026-04-08: Make Unity wall markers and procedural wall points snap immediately to their grounded placement instead of inheriting the tracked-marker movement smoothing.
2026-04-08: Stop tinting Unity prefab-backed markers on first spawn by applying the same remove-color rule during creation that later update frames already use.
2026-04-08: Make Unity wall control points use each marker's final raycast-snapped placement position so procedural wall segments inherit the same grounded Y as their source markers.
2026-04-08: Replace Unity wall prefab segments with procedural prism meshes built directly from marker pairs, wall thickness, and wall height, anchored to the higher marker Y to avoid wall flipping.
2026-04-08: Stabilize Unity wall segment elevation by anchoring each wall to the higher of its two marker Y positions instead of averaging endpoint heights.
2026-04-08: Remove the legacy Unity per-tag prefab override list so prefab and wall behavior are configured only through the new construction bindings inspector path.
2026-04-08: Add extensible Unity construction bindings so marker indexes can drive either prefab spawning or ordered wall generation, with inspector-controlled wall prefab, height, thickness, and local axis directions for future construction types.
2026-04-08: Fix Unity marker placement for prefab buildings by snapping with pivot-aware top and bottom bounds offsets so base-pivoted meshes land on terrain instead of hovering above it.
2026-04-04: Audit the repo docs against the current AprilTag implementation, add the missing registration-coords route details, and align the documented test command with the working uv invocation.
2026-04-04: Shorten the README into a quick-start overview and move the detailed runtime and service instructions into docs/runtime-reference.md.
2026-04-01: Clamp Unity tracked-marker movement to 5 world units per detection update before snapping onto the collider below, instead of teleporting directly to each new decode position.
2026-04-01: Add a Unity renderer flag that skips tag-color tinting for prefab-spawned markers by default while keeping primitive cubes colorized.
2026-04-01: Let Unity MarkerDetectionRenderer spawn per-tag prefabs from an index-to-prefab override list, falling back to the default cube when no override exists.
2026-04-01: Overhaul Unity marker tracking to upsert one cube per tag id, snap placements immediately onto the first collider below, remove fall physics, and expire unseen tags after 3 seconds.
2026-04-01: Prevent Unity tracked markers from expiring between websocket updates by disabling zero-second expiry and clamping short lifetimes above the send interval.
2026-04-01: Settle Unity marker cubes instantly onto the first ground or blocking collider below their offset spawn position instead of pinning them at the offset height.
2026-04-01: Spawn Unity marker cubes directly on the terrain and stop replaying the above-ground drop path so tracked tags land immediately.
2026-04-01: Restore the Unity MarkerDetectionRenderer runtime detection counters in the inspector for live debugging.
2026-04-01: Track Unity marker blocks by AprilTag value, keep only the first same-ID detection per frame, assign stable per-tag colors, and expire missing tags after a delay instead of respawning nearby cubes.
2026-03-25: Rename all remaining QR-prefixed Unity files/classes and Python QRCode types to Marker-prefixed equivalents.
2026-03-25: Remove the redundant CLAUDE.md instructions file and keep AGENTS.md as the single repo guidance source.
2026-03-25: Rename the Unity renderer component to MarkerDetectionRenderer and strip inspector grouping attributes from the script.
2026-03-25: Remove QR detection support, make the stack AprilTag-only, and replace the bundled QR sample with an AprilTag test image.
2026-03-25: Use the normal `detect` websocket payload for Unity debug-image requests so manually loaded textures bypass Unity flip processing.
2026-03-25: Normalize QR and AprilTag bounding boxes from decoded quad geometry so both detector modes project detections consistently.
2026-03-25: Add a synthetic QR-vs-AprilTag bbox regression test that holds detector placement deltas within a fixed epsilon.
