import signal
import sys
import time

import cv2
import numpy as np

from qr_to_pos import Camera, QRCodeProcessor


class InteractiveCLI:
    """Interactive command line interface for camera and QR code processing."""
    
    def __init__(self):
        self.camera: Camera | None = None
        self.processor: QRCodeProcessor | None = None
        self.last_detection = None
        self.latest_detections: list = []  # Store latest detections for visualization
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\nShutting down...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources."""
        if self.processor:
            self.processor.stop()
            self.processor = None
        if self.camera:
            self.camera.stop()
            self.camera = None
    
    def print_menu(self):
        """Print the main menu."""
        print("\n" + "=" * 50)
        print("QR Code Detection - Interactive CLI")
        print("=" * 50)
        print("Camera:", "Running" if self.camera and self.camera._running else "Stopped")
        print("Processor:", "Running" if self.processor and self.processor._running else "Stopped")
        print("Visualization:", "Running" if self.camera and self.camera._visualization_running else "Stopped")
        if self.last_detection:
            print(f"Last QR Code: {self.last_detection.data}")
        print("-" * 50)
        print("Commands:")
        print("  1. Start camera and processor with visualization")
        print("  2. Visualize camera only")
        print("  3. Stop camera")
        print("  4. Exit")
        print("=" * 50)
    
    def start_camera_and_processor(self):
        """Start camera and processor with visualization."""
        try:
            print("Initializing RealSense camera...")
            self.camera = Camera(width=1280, height=720, target_fps=30)
            self.camera.start()
            print("Camera started successfully!")
            
            # Wait a bit to ensure camera is capturing
            time.sleep(0.5)
            
            # Test frame capture
            frame = self.camera.get_latest_frame()
            if frame:
                print(f"Camera is capturing frames (resolution: {frame.data.shape[1]}x{frame.data.shape[0]})")
            else:
                print("Warning: Camera started but no frames captured yet.")
            
            print("Starting QR code processor...")
            self.processor = QRCodeProcessor(
                camera=self.camera,
                min_interval=0.1,
                model_size='s'
            )
            
            # Register callback to print detections and store for visualization
            def on_qr_detected(result):
                qr_codes = result.result
                # Store all detections for visualization
                self.latest_detections = qr_codes
                # Store the first one as last_detection for menu display
                self.last_detection = qr_codes[0] if qr_codes else None
                
                print(f"\n[QR Detected] Found {len(qr_codes)} QR code(s):")
                for i, qr in enumerate(qr_codes, 1):
                    print(f"  QR {i}:")
                    print(f"    Data: {qr.data}")
                    if qr.bbox:
                        print(f"    Bounding box: {qr.bbox}")
                    if qr.confidence:
                        print(f"    Confidence: {qr.confidence:.2f}")
                print(f"  Frame: {result.frame_index}, Time: {result.processing_time:.3f}s")
            
            self.processor.on_result(on_qr_detected)
            self.processor.start()
            print("QR code processor started!")
            print("\n" + "=" * 50)
            print("System is running. Press 'q' in the window or close it to return to menu.")
            print("=" * 50)
            
            # Start visualization loop
            self._visualization_loop()
            
        except Exception as e:
            print(f"Error starting system: {e}")
            self.cleanup()
            raise
    
    def _visualization_loop(self):
        """Visualization loop showing raw image on left and annotated image on right."""
        window_name = 'QR Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.camera and self.camera._running and self.processor and self.processor._running:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.033)
                    continue
                
                # Get raw image
                raw_image = frame.data.copy()
                
                # Create annotated image (copy of raw)
                annotated_image = raw_image.copy()
                
                # Draw boxes on annotated image
                for qr in self.latest_detections:
                    if qr.bbox:
                        x1, y1, x2, y2 = qr.bbox
                        # Draw rectangle
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with data and confidence
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
                            cv2.FILLED
                        )
                        cv2.putText(
                            annotated_image,
                            label,
                            (x1, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2
                        )
                
                # Combine images side by side
                combined_image = np.hstack((raw_image, annotated_image))
                
                # Display the combined image
                cv2.imshow(window_name, combined_image)
                
                # Check for window close or 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                time.sleep(0.033)  # ~30 FPS display rate
        finally:
            cv2.destroyWindow(window_name)
            # Clear detections when visualization stops
            self.latest_detections = []
    
    def start_visualization(self):
        """Start camera-only visualization."""
        # Automatically start camera if not running
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
        
        if self.camera._visualization_running:
            print("Visualization is already running!")
            return
        
        try:
            self.camera.start_visualization()
        except Exception as e:
            print(f"Error starting visualization: {e}")
    
    def stop_visualization(self):
        """Stop camera-only visualization."""
        if self.camera:
            self.camera.stop_visualization()
    
    def stop_camera(self):
        """Stop the camera and processor."""
        if not self.camera:
            print("Camera is not running!")
            return
        
        if not self.camera._running:
            print("Camera is already stopped!")
            return
        
        # Stop processor first if running
        if self.processor and self.processor._running:
            print("Stopping processor...")
            self.processor.stop()
            self.processor = None
        
        # Stop visualization if running
        if self.camera._visualization_running:
            print("Stopping visualization...")
            self.camera.stop_visualization()
        
        # Stop camera
        print("Stopping camera...")
        self.camera.stop()
        self.camera = None
        print("Camera stopped successfully!")
    
    def run(self):
        """Run the interactive CLI."""
        print("Welcome to QR Code Detection System!")
        print("Press Ctrl+C at any time to exit gracefully.\n")
        
        try:
            while True:
                self.print_menu()
                
                try:
                    choice = input("\nEnter command (1-4): ").strip()
                    
                    if choice == '1':
                        self.start_camera_and_processor()
                        # Visualization loop blocks until window is closed or 'q' is pressed
                        # After it returns, cleanup and return to menu
                        self.cleanup()
                    elif choice == '2':
                        print("\nVisualization running. Press 'q' in the window or close it to return to menu...")
                        self.start_visualization()
                        # start_visualization() blocks until stopped, so we're back at menu now
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
