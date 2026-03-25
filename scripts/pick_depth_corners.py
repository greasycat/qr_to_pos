"""
Interactive depth corner picker.

Opens a depth TXT, visualizes it, optionally seeds the view with auto-detected
corners, and lets you confirm or replace the 4 box corners with mouse clicks.

Controls:
  Left-click  — place points in order: TL, TR, BR, BL
  Right-click — reset all points
  Close window when 4 points are correct

Usage:
    uv run python scripts/pick_depth_corners.py \
        --depth assets/registration/1.txt

    uv run python scripts/pick_depth_corners.py \
        --depth assets/registration/1.txt \
        --out assets/registration/depth_corners.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from marker_to_pos.registration import (
    detect_box_corners_depth,
    load_depth_map,
    pick_depth_corners_interactive,
    visualize_depth,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick 4 depth-image corners interactively.")
    parser.add_argument("--depth", default="assets/registration/1.txt", help="Path to depth TXT")
    parser.add_argument("--out", help="Optional output path for .npy corner coordinates")
    args = parser.parse_args()

    depth = load_depth_map(args.depth)
    depth_vis = visualize_depth(depth)

    suggested = detect_box_corners_depth(depth)
    if suggested is not None:
        print("Loaded auto-detected depth corners as the starting suggestion.")
        print("Right-click to clear them and re-pick manually if needed.")
        print(suggested)
    else:
        print("No auto-detected depth corners found. Please click 4 corners manually.")

    corners = pick_depth_corners_interactive(depth_vis, initial_pts=suggested)

    print("\nConfirmed depth corners:")
    print(corners)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, corners)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
