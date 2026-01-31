import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class Frame:
    data: np.ndarray
    timestamp: float
    index: int


class Camera:
    
    def __init__(
        self,
        device_id: int = 0,
        target_fps: float | None = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.target_fps = target_fps
        self.width = width
        self.height = height
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Check for RGB camera
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            raise RuntimeError("The demo requires Depth camera with Color sensor")
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        
        self._latest_frame: Frame | None = None
        self._frame_lock = threading.Lock()
        self._frame_index = 0
        self._capture_thread: threading.Thread | None = None
        self._running = False
        self._callbacks: list[Callable[[Frame], None]] = []
        self._callback_lock = threading.Lock()
        
        # Visualization state
        self._visualization_running = False
        self._window_name = 'Camera View'
    
    def start(self) -> None:
        """Start the capture thread."""
        if self._running:
            return
        
        self.pipeline.start(self.config)
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def stop(self, timeout: float = 2.0) -> None:
        """Stop the capture thread."""
        self.stop_visualization()
        if not self._running:
            return
        
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=timeout)
        self.pipeline.stop()
    
    def get_latest_frame(self) -> Frame | None:
        """Get the most recent frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame
    
    def on_frame(self, callback: Callable[[Frame], None]) -> None:
        """Register a callback to be called on each new frame."""
        with self._callback_lock:
            self._callbacks.append(callback)
    
    def _capture_loop(self) -> None:
        """Internal capture loop (runs in thread)."""
        frame_time = 1.0 / self.target_fps if self.target_fps else 0.0
        last_frame_time = time.time()
        
        while self._running:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Rate limiting
                if self.target_fps:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                    last_frame_time = time.time()
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Create frame object
                frame = Frame(
                    data=color_image,
                    timestamp=time.time(),
                    index=self._frame_index
                )
                self._frame_index += 1
                
                # Update latest frame
                with self._frame_lock:
                    self._latest_frame = frame
                
                # Call callbacks
                with self._callback_lock:
                    for callback in self._callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            print(f"Error in frame callback: {e}")
                            
            except Exception as e:
                if self._running:
                    print(f"Error in capture loop: {e}")
                    time.sleep(0.1)
    
    def start_visualization(self, window_name: str = 'Camera View') -> None:
        if not self._running:
            raise RuntimeError("Camera must be started before starting visualization")
        
        if self._visualization_running:
            return
        
        self._window_name = window_name
        self._visualization_running = True
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self._visualization_running:
                frame = self.get_latest_frame()
                if frame is not None:
                    cv2.imshow(self._window_name, frame.data)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._visualization_running = False
                    break
                    
                time.sleep(0.033)  # ~30 FPS display rate
        finally:
            cv2.destroyWindow(self._window_name)
            self._visualization_running = False
    
    def stop_visualization(self) -> None:
        """Stop camera-only visualization."""
        if not self._visualization_running:
            return
        
        self._visualization_running = False
        cv2.destroyWindow(self._window_name)
    
    def __enter__(self) -> "Camera":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
