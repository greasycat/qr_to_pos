from .camera import Camera, Frame
from .inspection import InteractiveCLI, show_camera, show_detections
from .processor import MarkerDetection, MarkerDetectionProcessor, ProcessingResult
from .registration import (
    compute_homography,
    detect_box_corners_color,
    detect_box_corners_depth,
    load_depth_map,
    load_registration,
    map_bbox_to_depth,
    pick_depth_corners_interactive,
    save_registration,
    visualize_depth,
)
def __getattr__(name: str):
    if name == "DetectionServer":
        from .server import DetectionServer
        return DetectionServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Camera",
    "DetectionServer",
    "Frame",
    "InteractiveCLI",
    "ProcessingResult",
    "MarkerDetection",
    "MarkerDetectionProcessor",
    "compute_homography",
    "detect_box_corners_color",
    "detect_box_corners_depth",
    "load_depth_map",
    "load_registration",
    "map_bbox_to_depth",
    "pick_depth_corners_interactive",
    "save_registration",
    "show_camera",
    "show_detections",
    "visualize_depth",
]
