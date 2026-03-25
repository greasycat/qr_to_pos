import signal
import sys
import time
from typing import Callable

import cv2
import numpy as np

from .camera import Camera
from .processor import QRCode, QRCodeProcessor


def show_camera(camera: Camera, window_name: str = 'Camera View') -> None:
    """Display live camera feed. Blocks until 'q' pressed or window closed."""
    if not camera._running:
        raise RuntimeError("Camera must be started before visualization")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while camera._running:
            frame = camera.get_latest_frame()
            if frame is not None:
                cv2.imshow(window_name, frame.data)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            time.sleep(0.033)
    finally:
        cv2.destroyWindow(window_name)


def show_detections(
    camera: Camera,
    get_detections: Callable[[], list[QRCode]],
    window_name: str = "AprilTag Detection",
) -> None:
    """Side-by-side raw/annotated view with marker bounding boxes. Blocks until 'q' or window closed."""
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while camera._running:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.033)
                continue

            raw_image = frame.data.copy()
            annotated_image = raw_image.copy()

            for qr in get_detections():
                if qr.bbox:
                    x1, y1, x2, y2 = qr.bbox
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = qr.data[:20] + '...' if len(qr.data) > 20 else qr.data
                    if qr.confidence is not None:
                        label = f'{label} ({qr.confidence:.2f})'

                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_y = max(y1, label_size[1] + 10)
                    cv2.rectangle(
                        annotated_image,
                        (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )

            combined = np.hstack((raw_image, annotated_image))
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            time.sleep(0.033)
    finally:
        cv2.destroyWindow(window_name)


class InteractiveCLI:
    """Interactive command line interface for camera and AprilTag processing."""

    def __init__(self):
        self.camera: Camera | None = None
        self.processor: QRCodeProcessor | None = None
        self.last_detection = None
        self.latest_detections: list = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\nShutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if self.processor:
            self.processor.stop()
            self.processor = None
        if self.camera:
            self.camera.stop()
            self.camera = None

    def print_menu(self):
        print("\n" + "=" * 50)
        print("AprilTag Detection - Interactive CLI")
        print("=" * 50)
        print("Camera:", "Running" if self.camera and self.camera._running else "Stopped")
        print("Processor:", "Running" if self.processor and self.processor._running else "Stopped")
        if self.last_detection:
            print(f"Last Marker: {self.last_detection.data}")
        print("-" * 50)
        print("Commands:")
        print("  1. Start camera and processor with visualization")
        print("  2. Visualize camera only")
        print("  3. Stop camera")
        print("  4. Exit")
        print("=" * 50)

    def start_camera_and_processor(self):
        try:
            print("Initializing RealSense camera...")
            self.camera = Camera(width=1280, height=720, target_fps=30)
            self.camera.start()
            print("Camera started successfully!")

            time.sleep(0.5)

            frame = self.camera.get_latest_frame()
            if frame:
                print(f"Camera is capturing frames (resolution: {frame.data.shape[1]}x{frame.data.shape[0]})")
            else:
                print("Warning: Camera started but no frames captured yet.")

            print("Starting AprilTag processor...")
            self.processor = QRCodeProcessor(
                camera=self.camera,
                min_interval=0.1,
                model_size="s",
            )

            def on_qr_detected(result):
                qr_codes = result.result
                self.latest_detections = qr_codes
                self.last_detection = qr_codes[0] if qr_codes else None

                print(f"\n[Marker Detected] Found {len(qr_codes)} marker(s):")
                for i, qr in enumerate(qr_codes, 1):
                    print(f"  Marker {i}:")
                    print(f"    Data: {qr.data}")
                    if qr.bbox:
                        print(f"    Bounding box: {qr.bbox}")
                    if qr.confidence:
                        print(f"    Confidence: {qr.confidence:.2f}")
                print(f"  Frame: {result.frame_index}, Time: {result.processing_time:.3f}s")

            self.processor.on_result(on_qr_detected)
            self.processor.start()
            print("AprilTag processor started!")
            print("\n" + "=" * 50)
            print("System is running. Press 'q' in the window or close it to return to menu.")
            print("=" * 50)

            show_detections(self.camera, lambda: self.latest_detections)
            self.latest_detections = []

        except Exception as e:
            print(f"Error starting system: {e}")
            self.cleanup()
            raise

    def start_visualization(self):
        if not self.camera or not self.camera._running:
            print("Starting camera...")
            try:
                self.camera = Camera(width=1280, height=720, target_fps=30)
                self.camera.start()
                time.sleep(0.5)
                print("Camera started!")
            except Exception as e:
                print(f"Error starting camera: {e}")
                return

        try:
            show_camera(self.camera)
        except Exception as e:
            print(f"Error starting visualization: {e}")

    def stop_camera(self):
        if not self.camera:
            print("Camera is not running!")
            return

        if not self.camera._running:
            print("Camera is already stopped!")
            return

        if self.processor and self.processor._running:
            print("Stopping processor...")
            self.processor.stop()
            self.processor = None

        print("Stopping camera...")
        self.camera.stop()
        self.camera = None
        print("Camera stopped successfully!")

    def run(self):
        print("Welcome to the AprilTag Detection System!")
        print("Press Ctrl+C at any time to exit gracefully.\n")

        try:
            while True:
                self.print_menu()

                try:
                    choice = input("\nEnter command (1-4): ").strip()

                    if choice == '1':
                        self.start_camera_and_processor()
                        self.cleanup()
                    elif choice == '2':
                        print("\nVisualization running. Press 'q' in the window or close it to return to menu...")
                        self.start_visualization()
                        if self.camera and not self.camera._running:
                            self.camera = None
                    elif choice == '3':
                        self.stop_camera()
                    elif choice == '4':
                        print("Exiting...")
                        self.cleanup()
                        break
                    else:
                        print("Invalid choice! Please enter a number between 1-4.")

                    time.sleep(0.1)

                except KeyboardInterrupt:
                    print("\n\nReturning to menu...")
                    self.cleanup()
                    continue
                except EOFError:
                    print("\n\nExiting...")
                    self.cleanup()
                    break

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.cleanup()
        except EOFError:
            print("\n\nShutting down...")
            self.cleanup()
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup()
            raise


def main():
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
