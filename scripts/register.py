"""
One-shot CLI: detect or pick box corners on a color+depth pair, save homography.

Usage:
    uv run python scripts/register.py \
        --color assets/registration/1.png \
        --depth assets/registration/1.txt \
        --out   assets/registration/homography.npy

    # Force manual corner picking
    uv run python scripts/register.py \
        --color assets/registration/1.png \
        --depth assets/registration/1.txt \
        --out   assets/registration/homography.npy \
        --manual
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from marker_to_pos.registration import (
    compute_homography,
    detect_box_corners_color,
    detect_box_corners_depth,
    load_depth_map,
    pick_corners_interactive,
    save_registration,
    visualize_depth,
)

_LABELS = ["TL", "TR", "BR", "BL"]
_COLORS = ["lime", "orange", "red", "blue"]


def _overlay_corners(ax, pts: np.ndarray, title: str) -> None:
    ax.set_title(title)
    for i, (x, y) in enumerate(pts):
        ax.plot(x, y, "o", color=_COLORS[i], markersize=10)
        ax.text(x + 8, y - 8, _LABELS[i], color=_COLORS[i], fontsize=12, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register color→depth via 4-corner homography.")
    parser.add_argument("--color", required=True, help="Path to color PNG")
    parser.add_argument("--depth", required=True, help="Path to depth TXT")
    parser.add_argument("--out", required=True, help="Output path for homography .npy")
    parser.add_argument("--manual", action="store_true", help="Force manual corner picking")
    args = parser.parse_args()

    color_img = cv2.imread(args.color)
    if color_img is None:
        sys.exit(f"Cannot read color image: {args.color}")

    depth = load_depth_map(args.depth)
    depth_vis = visualize_depth(depth)

    color_pts = None
    depth_pts = None

    if not args.manual:
        print("Auto-detecting box corners...")
        color_pts = detect_box_corners_color(color_img)
        depth_pts = detect_box_corners_depth(depth)

        if color_pts is not None:
            print(f"  Color corners detected:\n{color_pts}")
        else:
            print("  Color auto-detection failed.")

        if depth_pts is not None:
            print(f"  Depth corners detected:\n{depth_pts}")
        else:
            print("  Depth auto-detection failed.")

    if color_pts is not None and depth_pts is not None:
        color_rgb = color_img[:, :, ::-1]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(color_rgb)
        _overlay_corners(axes[0], color_pts, "Color frame — auto-detected corners")
        axes[1].imshow(depth_vis, cmap="viridis")
        _overlay_corners(axes[1], depth_pts, "Depth frame — auto-detected corners")
        plt.tight_layout()
        print("\nClose the preview window to accept and save. Re-run with --manual to pick manually.")
        plt.show()
    else:
        print("\nFalling back to manual corner picking.")
        print("Pick 4 corners (TL → TR → BR → BL) in each window.")
        print("Left-click to place, right-click to reset, close window to confirm.")
        color_pts, depth_pts = pick_corners_interactive(
            color_img,
            depth_vis,
            color_initial_pts=color_pts,
            depth_initial_pts=depth_pts,
        )

    H = compute_homography(color_pts, depth_pts)
    print("\nHomography matrix:")
    print(H)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_registration(H, str(out_path))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
