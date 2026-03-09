"""
QR-to-depth registration: homography-based color→depth pixel mapping.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "registration" / "config.yml"


def _load_registration_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _load_depth_config() -> dict:
    return _load_registration_config()


def _load_depth_clip() -> tuple[float, float]:
    cfg = _load_registration_config()
    dc = cfg.get("depth_clip", {})
    return float(dc.get("min", 1.065)), float(dc.get("max", 1.235))

_LABELS = ["TL", "TR", "BR", "BL"]
_COLORS = ["lime", "orange", "red", "blue"]


def load_depth_map(txt_path: str) -> np.ndarray:
    """Parse depth TXT: skip header line, load tab-separated floats → (H, W) float32."""
    with open(txt_path) as f:
        lines = f.readlines()
    # First line is "Frame size: W x H"
    data_lines = [l for l in lines[1:] if l.strip()]
    rows = [list(map(float, l.strip().split("\t"))) for l in data_lines]
    return np.array(rows, dtype=np.float32)


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize non-zero depth to [0,1] float; zeros stay 0. Returns float32 array for matplotlib."""
    vis = np.zeros_like(depth, dtype=np.float32)
    mask = depth > 0
    if mask.any():
        dmin, dmax = depth[mask].min(), depth[mask].max()
        if dmax > dmin:
            vis[mask] = (depth[mask] - dmin) / (dmax - dmin)
        else:
            vis[mask] = 1.0
    return vis


def pick_points_interactive(
    title: str,
    img: np.ndarray,
    cmap: str | None = None,
    initial_pts: np.ndarray | None = None,
) -> np.ndarray:
    """
    Show one image and collect 4 points (TL, TR, BR, BL).
    Left click adds points until 4 are present. Right click resets all points.
    Close the window to confirm after 4 clicks.
    """
    pts = [] if initial_pts is None else [tuple(map(float, pt)) for pt in initial_pts]
    scatter_artists: list = []

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, cmap=cmap)
    ax.set_title(f"{title}\nLeft-click: place point | Right-click: reset | Close when done (need 4)")
    status_text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.5),
    )

    def _redraw() -> None:
        for artist in scatter_artists:
            artist.remove()
        scatter_artists.clear()
        for i, (x, y) in enumerate(pts):
            scatter_artists.append(ax.plot(x, y, "o", color=_COLORS[i], markersize=10)[0])
            scatter_artists.append(
                ax.text(x + 8, y - 8, _LABELS[i], color=_COLORS[i], fontsize=12, fontweight="bold")
            )
        suffix = " (suggested)" if initial_pts is not None and len(pts) == 4 else ""
        status_text.set_text(f"{len(pts)}/4 points{suffix}")
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 1 and len(pts) < 4:
            pts.append((event.xdata, event.ydata))
            _redraw()
        elif event.button == 3:
            pts.clear()
            _redraw()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    _redraw()
    plt.tight_layout()
    plt.show()

    if len(pts) != 4:
        raise RuntimeError(f"Expected 4 points, got {len(pts)}. Re-run and click exactly 4 corners.")
    return np.array(pts, dtype=np.float32)


def pick_color_corners_interactive(
    color_img: np.ndarray,
    initial_pts: np.ndarray | None = None,
) -> np.ndarray:
    """Show the color image and let the user confirm or replace 4 corners."""
    color_rgb = color_img[:, :, ::-1]
    return pick_points_interactive("Color frame", color_rgb, initial_pts=initial_pts)


def pick_depth_corners_interactive(
    depth_vis: np.ndarray,
    initial_pts: np.ndarray | None = None,
) -> np.ndarray:
    """Show the depth visualization and let the user confirm or replace 4 corners."""
    return pick_points_interactive("Depth frame", depth_vis, cmap="viridis", initial_pts=initial_pts)


def pick_corners_interactive(
    color_img: np.ndarray,
    depth_vis: np.ndarray,
    color_initial_pts: np.ndarray | None = None,
    depth_initial_pts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Show color then depth image; user confirms or clicks 4 corners (TL, TR, BR, BL).
    Right-click resets current image's points. Close each window to confirm.
    Returns (color_pts, depth_pts) each as float32 (4,2).
    """
    color_pts = pick_color_corners_interactive(color_img, initial_pts=color_initial_pts)
    depth_pts = pick_depth_corners_interactive(depth_vis, initial_pts=depth_initial_pts)
    return color_pts, depth_pts


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points [TL, TR, BR, BL] by sum/diff of coordinates."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex B in the triangle A-B-C, in degrees."""
    u = a - b
    v = c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-10:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def _pick_4_from_poly(poly_pts: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    """
    Slide a window of 4 consecutive vertices A,B,C,D over the polygon.
    Compute angles A-B-C, B-C-D, C-D-A and sum them.
    A rectangular corner set has sum ≈ 270°.
    Returns the best-matching window ordered as [TL,TR,BR,BL], or None if < 4 pts.
    """
    n = len(poly_pts)
    if n < 4:
        return None
    best_idx, best_diff = 0, float("inf")
    for i in range(n):
        A, B, C, D = (poly_pts[j % n] for j in (i, i + 1, i + 2, i + 3))
        a1, a2, a3 = _angle_at(A, B, C), _angle_at(B, C, D), _angle_at(C, D, A)
        total = a1 + a2 + a3
        diff = abs(total - 270.0) + abs(a1 - 90.0) + abs(a2 - 90.0) + abs(a3 - 90.0)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    i = best_idx
    quad = np.array([poly_pts[j % n] for j in (i, i + 1, i + 2, i + 3)], dtype=np.float32)
    return _order_corners(quad)


def detect_box_corners_color(
    color_img: np.ndarray,
    min_area_ratio: float = 0.05,
    canny_lo: int = 50,
    canny_hi: int = 150,
) -> np.ndarray | None:
    """
    Find the largest 4-sided contour in the color image.
    Uses HSV saturation thresholding (Otsu) to separate the colored box from
    a grey floor; falls back to Canny edges if saturation gives no result.
    Returns (4,2) float32 [TL,TR,BR,BL] or None if no quad found above min_area_ratio.
    """
    h, w = color_img.shape[:2]
    min_area = min_area_ratio * h * w
    def _largest_quad(binary: np.ndarray) -> tuple[np.ndarray | None, float]:
        kernel = np.ones((11, 11), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in [0.02, 0.03, 0.05, 0.08]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and area > best_area:
                    best = approx.reshape(4, 2).astype(np.float32)
                    best_area = area
                    break
        return best, best_area

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, _ = _largest_quad(sat_thresh)
    if best is not None:
        return _order_corners(best)

    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, _ = None, 0.0
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and area > best_area:
                best = approx.reshape(4, 2).astype(np.float32)
                best_area = area
                break

    return _order_corners(best) if best is not None else None


def detect_box_corners_depth(
    depth: np.ndarray,
    d_lo: float | None = None,
    d_hi: float | None = None,
    morph_kernel: int | None = None,
    erode_kernel: int | None = None,
    min_component_area: int | None = None,
    eps: float | None = None,
) -> np.ndarray | None:
    """
    Find box outline in depth map via:
      1. Clip+invert depth → Otsu threshold
      2. Morphological close+open (morph_kernel) to fill holes and remove noise
      3. Small open (3×3) to clean tiny specks
      4. Erode (erode_kernel) to break connections between regions
      5. Connected components → keep rightmost component by centroid
      6. Dilate back → approxPolyDP (eps); if not 4 vertices, fallback to minAreaRect

    All parameters default to values in assets/registration/config.yml.
    Tune them interactively with scripts/tune_depth_contour.py.
    Returns (4,2) float32 [TL,TR,BR,BL] or None.
    """
    cfg = _load_depth_config()
    dc = cfg.get("depth_clip", {})
    cc = cfg.get("depth_contour", {})

    if d_lo is None:
        d_lo = float(dc.get("min", 1.065))
    if d_hi is None:
        d_hi = float(dc.get("max", 1.235))
    if morph_kernel is None:
        morph_kernel = int(cc.get("morph_kernel", 7))
    if erode_kernel is None:
        erode_kernel = int(cc.get("erode_kernel", 9))
    if min_component_area is None:
        min_component_area = int(cc.get("min_component_area", 500))
    if eps is None:
        eps = float(cc.get("eps", 0.08))

    valid_mask = depth > 0
    if not valid_mask.any():
        return None

    clipped = np.clip(depth, d_lo, d_hi)
    clipped[~valid_mask] = d_hi  # treat invalid pixels as far (dark after inversion)
    inverted = d_hi - clipped    # close/elevated → bright

    norm = cv2.normalize(inverted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes and remove large noise
    k = max(1, morph_kernel | 1)  # force odd
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Clean tiny specks
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Break connections between regions
    ek = max(1, erode_kernel | 1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek))
    separated = cv2.erode(bw, erode_k, iterations=1)

    # Connected components: keep the component whose centroid is farthest right
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(separated)
    best_label = None
    best_x = -1.0
    for i in range(1, num_labels):  # skip background (0)
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            continue
        cx = float(centroids[i][0])
        if cx > best_x:
            best_x = cx
            best_label = i

    if best_label is None:
        return None

    right_mask = np.zeros_like(bw)
    right_mask[labels == best_label] = 255

    # Recover shape lost to erosion
    right_mask = cv2.dilate(right_mask, erode_k, iterations=1)

    contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps * peri, True)
    poly_pts = approx.reshape(-1, 2).astype(np.float32)
    corners = _pick_4_from_poly(poly_pts, depth.shape)
    if corners is not None:
        return corners
    # fallback: minimum area rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return _order_corners(np.float32(box))


def compute_homography(color_pts: np.ndarray, depth_pts: np.ndarray) -> np.ndarray:
    """cv2.getPerspectiveTransform(color_pts, depth_pts) → 3×3 float64."""
    return cv2.getPerspectiveTransform(color_pts, depth_pts)


def save_registration(H: np.ndarray, path: str) -> None:
    np.save(path, H)


def load_registration(path: str) -> np.ndarray:
    return np.load(path)


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project Nx2 color-frame points into depth-frame coordinates via homography."""
    src = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H)
    return dst.reshape(-1, 2)


def transform_bbox_to_depth(bbox: list[float] | tuple[float, float, float, float], H: np.ndarray) -> np.ndarray:
    """Project bbox corners [TL, TR, BR, BL] from color pixels into depth pixels."""
    x1, y1, x2, y2 = map(float, bbox)
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )
    return transform_points(corners, H)


def map_bbox_to_depth(
    bbox: list[float], H: np.ndarray, depth: np.ndarray
) -> float | None:
    """
    bbox = [x1, y1, x2, y2] in color-frame pixels.
    Project the center through H, look up depth[row, col].
    Returns None if out-of-bounds or depth == 0.
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    dst = transform_points(np.array([[cx, cy]], dtype=np.float64), H)
    col = int(round(float(dst[0, 0])))
    row = int(round(float(dst[0, 1])))
    h, w = depth.shape
    if row < 0 or row >= h or col < 0 or col >= w:
        return None
    val = float(depth[row, col])
    return val if val > 0 else None
