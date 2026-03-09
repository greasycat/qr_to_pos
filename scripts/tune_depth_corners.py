"""
Interactive depth contour parameter tuner.

Four panels show each processing stage in real-time as you adjust sliders:
  1. Clipped + inverted depth (normalized) — effect of clip min/max
  2. After close/open/speck removal (bw)   — effect of morph kernel
  3. Selected component mask (right_mask)  — effect of erode kernel + min area
  4. Detected corners on depth image       — final result

Sliders:
  Clip min / Clip max   — depth range to isolate the box surface
  Morph kernel          — close+open kernel size (odd, 1–21)
  Erode kernel          — erosion kernel to break connections (odd, 1–21)
  Min component area    — minimum CC area in pixels to consider

Close the window to save all parameters to assets/registration/config.yml.

Usage:
    uv run python scripts/tune_depth_contour.py \\
        --depth assets/registration/1.txt
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Slider

sys.path.insert(0, str(Path(__file__).parent.parent))

from qr_to_pos.registration import load_depth_map, visualize_depth

_CONFIG_PATH = Path(__file__).parent.parent / "assets" / "registration" / "config.yml"
_CORNER_LABELS = ["TL", "TR", "BR", "BL"]
_CORNER_COLORS = ["lime", "orange", "red", "blue"]


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-10:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def _pick_4_from_poly(
    poly_pts: np.ndarray, img_shape: tuple
) -> tuple[np.ndarray | None, float, float, float, float]:
    """
    Slide window of 4 consecutive vertices A,B,C,D.
    Score = |sum−270| + |a1−90| + |a2−90| + |a3−90|  (lower is better).
    Returns (corners or None, a1, a2, a3, sum).
    """
    n = len(poly_pts)
    if n < 4:
        return None, 0.0, 0.0, 0.0, 0.0
    best_idx, best_diff = 0, float("inf")
    best_angles = (0.0, 0.0, 0.0)
    for i in range(n):
        A, B, C, D = (poly_pts[j % n] for j in (i, i + 1, i + 2, i + 3))
        a1, a2, a3 = _angle_at(A, B, C), _angle_at(B, C, D), _angle_at(C, D, A)
        total = a1 + a2 + a3
        diff = abs(total - 270.0) + abs(a1 - 90.0) + abs(a2 - 90.0) + abs(a3 - 90.0)
        if diff < best_diff:
            best_diff, best_angles, best_idx = diff, (a1, a2, a3), i
    i = best_idx
    quad = np.array([poly_pts[j % n] for j in (i, i + 1, i + 2, i + 3)], dtype=np.float32)
    a1, a2, a3 = best_angles
    return _order_corners(quad), a1, a2, a3, a1 + a2 + a3


def _run_pipeline(
    depth: np.ndarray,
    d_lo: float,
    d_hi: float,
    morph_kernel: int,
    erode_kernel: int,
    min_component_area: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float, float, float, float]:
    """
    Returns (norm, bw, right_mask, poly_pts, corners, a1, a2, a3, angle_sum).
      a1/a2/a3   : individual angles at B, C, D of best window (target ≈ 90° each)
      angle_sum  : a1+a2+a3 (target ≈ 270°)
    """
    valid_mask = depth > 0
    clipped = np.clip(depth, d_lo, d_hi)
    clipped[~valid_mask] = d_hi
    inverted = d_hi - clipped

    norm = cv2.normalize(inverted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(1, int(morph_kernel) | 1)
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ek = max(1, int(erode_kernel) | 1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (ek, ek))
    separated = cv2.erode(bw, erode_k, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(separated)
    best_label = None
    best_x = -1.0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component_area:
            continue
        cx = float(centroids[i][0])
        if cx > best_x:
            best_x = cx
            best_label = i

    right_mask = np.zeros_like(bw)
    poly_pts: np.ndarray | None = None
    corners: np.ndarray | None = None
    a1 = a2 = a3 = angle_sum = 0.0

    if best_label is not None:
        right_mask[labels == best_label] = 255
        right_mask = cv2.dilate(right_mask, erode_k, iterations=1)
        contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            poly_pts = approx.reshape(-1, 2).astype(np.float32)
            corners, a1, a2, a3, angle_sum = _pick_4_from_poly(poly_pts, depth.shape)
            if corners is None:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                corners = _order_corners(np.float32(box))

    return norm, bw, right_mask, poly_pts, corners, a1, a2, a3, angle_sum


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune depth contour detection parameters.")
    parser.add_argument("--depth", default="assets/registration/1.txt", help="Path to depth TXT")
    args = parser.parse_args()

    depth = load_depth_map(args.depth)
    depth_vis = visualize_depth(depth)
    valid = depth[depth > 0]
    abs_min = float(valid.min())
    abs_max = float(valid.max())

    cfg = _load_config()
    dc = cfg.get("depth_clip", {})
    cc = cfg.get("depth_contour", {})
    init_lo = float(dc.get("min", np.percentile(valid, 1)))
    init_hi = float(dc.get("max", np.percentile(valid, 98)))
    init_morph = int(cc.get("morph_kernel", 7))
    init_erode = int(cc.get("erode_kernel", 9))
    init_area = int(cc.get("min_component_area", 500))
    init_eps = float(cc.get("eps", 0.08))

    # ── Layout: 4 panels ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.35, wspace=0.05)

    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)
    for ax in (ax1, ax2, ax3, ax4):
        ax.axis("off")

    norm0, bw0, mask0, poly0, corners0, a1_0, a2_0, a3_0, angle0 = _run_pipeline(
        depth, init_lo, init_hi, init_morph, init_erode, init_area, init_eps
    )

    im1 = ax1.imshow(norm0, cmap="viridis", vmin=0, vmax=255)
    ax1.set_title("1. Clipped+inverted\n(input to Otsu)", fontsize=9)

    im2 = ax2.imshow(bw0, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("2. Close+open+speck\nremoval (bw)", fontsize=9)

    im3 = ax3.imshow(mask0, cmap="gray", vmin=0, vmax=255)
    ax3.set_title("3. approxPolyDP vertices\non component mask", fontsize=9)

    im4 = ax4.imshow(depth_vis, cmap="viridis")
    ax4.set_title("4. Detected corners", fontsize=9)

    poly_artists: list = []
    poly_status = ax3.text(
        0.02, 0.97, "",
        transform=ax3.transAxes, va="top", color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6),
    )
    corner_artists: list = []
    corner_status = ax4.text(
        0.02, 0.97, "",
        transform=ax4.transAxes, va="top", color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6),
    )

    # Cycle of distinct colors for polygon vertices
    _POLY_COLORS = [
        "red", "lime", "deepskyblue", "yellow", "magenta", "cyan",
        "orange", "white", "hotpink", "springgreen",
    ]

    def _draw_poly(poly_pts):
        for a in poly_artists:
            a.remove()
        poly_artists.clear()
        if poly_pts is None or len(poly_pts) == 0:
            poly_status.set_text("No component")
            return
        n = len(poly_pts)
        # Draw polygon outline
        closed_pts = np.vstack([poly_pts, poly_pts[0]])
        line = ax3.plot(closed_pts[:, 0], closed_pts[:, 1], "-",
                        color="white", linewidth=1.5, alpha=0.7)[0]
        poly_artists.append(line)
        # Draw each vertex with index
        for i, (x, y) in enumerate(poly_pts):
            color = _POLY_COLORS[i % len(_POLY_COLORS)]
            dot = ax3.plot(x, y, "o", color=color, markersize=8)[0]
            lbl = ax3.text(x + 5, y - 5, str(i), color=color, fontsize=9, fontweight="bold")
            poly_artists.extend([dot, lbl])
        poly_status.set_text(f"{n} vertices")

    def _draw_corners(corners, a1=0.0, a2=0.0, a3=0.0, angle_sum=0.0):
        for a in corner_artists:
            a.remove()
        corner_artists.clear()
        if corners is not None:
            for i, (x, y) in enumerate(corners):
                s = ax4.plot(x, y, "o", color=_CORNER_COLORS[i], markersize=10)[0]
                t = ax4.text(x + 6, y - 6, _CORNER_LABELS[i],
                             color=_CORNER_COLORS[i], fontsize=11, fontweight="bold")
                corner_artists.extend([s, t])
            corner_status.set_text(
                f"angles: {a1:.1f}° {a2:.1f}° {a3:.1f}°\nsum={angle_sum:.1f}°"
            )
        else:
            corner_status.set_text("No component found")

    _draw_poly(poly0)
    _draw_corners(corners0, a1_0, a2_0, a3_0, angle0)

    # ── Sliders ───────────────────────────────────────────────────────────────
    slider_specs = [
        # (left, bottom, label, vmin, vmax, valinit, valstep, color)
        (0.07, 0.28, "Clip min (m)",       abs_min, abs_max, init_lo,    0.005, "deepskyblue"),
        (0.07, 0.23, "Clip max (m)",       abs_min, abs_max, init_hi,    0.005, "tomato"),
        (0.07, 0.18, "Morph kernel",       1,       21,      init_morph, 2,     "gold"),
        (0.07, 0.13, "Erode kernel",       1,       21,      init_erode, 2,     "orange"),
        (0.07, 0.08, "Min component area", 50,      5000,    init_area,  50,    "violet"),
        (0.07, 0.03, "Poly eps",           0.005,   0.30,    init_eps,   0.005, "lightgreen"),
    ]

    sliders = []
    for left, bot, label, vmin, vmax, valinit, vstep, color in slider_specs:
        ax_s = fig.add_axes([left, bot, 0.86, 0.025])
        sl = Slider(ax_s, label, vmin, vmax, valinit=valinit, valstep=vstep, color=color)
        sliders.append(sl)

    sl_lo, sl_hi, sl_morph, sl_erode, sl_area, sl_eps = sliders

    def update(_):
        lo = sl_lo.val
        hi = sl_hi.val
        if lo >= hi:
            return
        norm, bw, mask, poly, corners, a1, a2, a3, angle_sum = _run_pipeline(
            depth, lo, hi, int(sl_morph.val), int(sl_erode.val), int(sl_area.val), sl_eps.val
        )
        im1.set_data(norm)
        im2.set_data(bw)
        im3.set_data(mask)
        _draw_poly(poly)
        _draw_corners(corners, a1, a2, a3, angle_sum)
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)

    plt.suptitle("Depth contour tuner — adjust sliders, close window to save", fontsize=12)
    plt.show()

    # ── Save ─────────────────────────────────────────────────────────────────
    lo = round(float(sl_lo.val), 3)
    hi = round(float(sl_hi.val), 3)
    morph = int(sl_morph.val) | 1
    erode = int(sl_erode.val) | 1
    area = int(sl_area.val)
    eps = round(float(sl_eps.val), 3)

    cfg["depth_clip"] = {"min": lo, "max": hi}
    cfg["depth_contour"] = {
        "morph_kernel": morph,
        "erode_kernel": erode,
        "min_component_area": area,
        "eps": eps,
    }

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to {_CONFIG_PATH}:")
    print(f"  depth_clip:         [{lo}, {hi}] m")
    print(f"  morph_kernel:       {morph}")
    print(f"  erode_kernel:       {erode}")
    print(f"  min_component_area: {area} px")
    print(f"  eps:                {eps}")


if __name__ == "__main__":
    main()
