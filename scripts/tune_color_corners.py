"""
Interactive color corner picker.

Opens a color PNG, optionally seeds the view with auto-detected corners, and
lets you confirm or replace the 4 box corners with mouse clicks.

Controls:
  Left-click  — place points in order: TL, TR, BR, BL
  Right-click — reset all points
  Close window when 4 points are correct

Usage:
    uv run python scripts/tune_color_corners.py \
        --color assets/registration/1.png

    uv run python scripts/tune_color_corners.py \
        --color assets/registration/1.png \
        --out assets/registration/color_corners.npy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from marker_to_pos.registration import detect_box_corners_color, pick_color_corners_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick 4 color-image corners interactively.")
    parser.add_argument("--color", default="assets/registration/1.png", help="Path to color PNG")
    parser.add_argument("--out", help="Optional output path for .npy corner coordinates")
    args = parser.parse_args()

    color_img = cv2.imread(args.color)
    if color_img is None:
        sys.exit(f"Cannot read color image: {args.color}")

    suggested = detect_box_corners_color(color_img)
    if suggested is not None:
        print("Loaded auto-detected corners as the starting suggestion.")
        print("Right-click to clear them and re-pick manually if needed.")
        print(suggested)
    else:
        print("No auto-detected corners found. Please click 4 corners manually.")

    corners = pick_color_corners_interactive(color_img, initial_pts=suggested)

    print("\nConfirmed color corners:")
    print(corners)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, corners)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
