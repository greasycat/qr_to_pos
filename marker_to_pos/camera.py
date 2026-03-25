import threading
import time
from dataclasses import dataclass
from typing import Callable

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
    
    def __enter__(self) -> "Camera":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
