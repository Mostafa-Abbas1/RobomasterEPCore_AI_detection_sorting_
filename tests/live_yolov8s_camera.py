"""
YOLOv8s Live Camera Test - RoboMaster EP
Connects to the RoboMaster EP robot and shows live object detection.

Controls:
- 'q' or ESC: Quit
- 's': Save screenshot
- '+'/'-': Adjust confidence threshold

Start: python tests/live_yolov8s_camera.py
"""

import cv2
import sys
import time
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robomaster import robot, camera as rm_camera
from src.robot_control import ThreadedCamera


class YOLOv8sRobotCameraTest:
    """Live camera test for YOLOv8s with RoboMaster EP camera"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initializes the camera test

        Args:
            confidence_threshold: Minimum confidence for detections (0.0 - 1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.ep_robot = None
        self.ep_camera = None
        self.threaded_cam = None
        self.running = False

        # Colors for different classes (BGR)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
        ]

    def load_model(self) -> bool:
        """Loads the YOLOv8s model"""
        try:
            from ultralytics import YOLO

            # Use model from project root
            model_path = PROJECT_ROOT / 'yolov8s.pt'
            print(f"Loading YOLOv8s model from {model_path}...")
            self.model = YOLO(str(model_path))
            print(f"Model loaded! Detectable classes: {len(self.model.names)}")
            return True

        except ImportError:
            print("ERROR: Ultralytics not installed!")
            print("Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def connect_robot(self) -> bool:
        """Connects to the RoboMaster EP robot"""
        try:
            print("Connecting to RoboMaster EP...")
            self.ep_robot = robot.Robot()
            self.ep_robot.initialize(conn_type="sta")

            version = self.ep_robot.get_version()
            print(f"Robot connected! Version: {version}")
            return True

        except Exception as e:
            print(f"ERROR: Could not connect to robot: {e}")
            return False

    def start_camera(self) -> bool:
        """Starts the RoboMaster camera"""
        try:
            print("Starting camera...")
            self.ep_camera = self.ep_robot.camera
            self.ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_720P)
            time.sleep(2)  # Wait until camera is ready

            # Threaded Camera for better performance
            self.threaded_cam = ThreadedCamera(self.ep_camera).start()
            time.sleep(1)

            print("Camera started! (720p)")
            return True

        except Exception as e:
            print(f"ERROR starting camera: {e}")
            return False

    def get_color(self, class_id: int) -> tuple:
        """Returns color for a class"""
        return self.colors[class_id % len(self.colors)]

    def draw_detections(self, frame, results) -> tuple:
        """
        Draws detections on the image

        Returns:
            tuple: (frame with drawings, detection count)
        """
        detection_count = 0

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Color for this class
                color = self.get_color(class_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with background
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 5, y1),
                    color,
                    -1
                )

                # Label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                detection_count += 1

        return frame, detection_count

    def draw_info_overlay(self, frame, fps: float, detection_count: int):
        """Draws info overlay on the image"""
        # Background for info
        cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (400, 100), (255, 255, 255), 1)

        # Info text
        cv2.putText(frame, "YOLOv8s Live Test - RoboMaster EP", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {detection_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.2f} (+/- to adjust)", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "'q'=Quit  's'=Screenshot", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def save_screenshot(self, frame):
        """Saves a screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def run(self):
        """Starts the live camera test"""
        print("\n" + "=" * 50)
        print("YOLOv8s Live Camera Test - RoboMaster EP")
        print("=" * 50 + "\n")

        # Load model
        if not self.load_model():
            return

        # Connect to robot
        if not self.connect_robot():
            return

        # Start camera
        if not self.start_camera():
            self.cleanup()
            return

        print("\nStarting live detection...")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  's': Save screenshot")
        print("  '+': Increase confidence")
        print("  '-': Decrease confidence")
        print()

        self.running = True
        fps = 0
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Get frame from RoboMaster camera
                frame = self.threaded_cam.read()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # YOLO detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)

                # Draw detections
                frame, detection_count = self.draw_detections(frame, results)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

                # Info overlay
                self.draw_info_overlay(frame, fps, detection_count)

                # Show frame
                cv2.imshow("YOLOv8s RoboMaster EP - Press 'q' to quit", frame)

                # Keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    print("\nQuitting...")
                    self.running = False

                elif key == ord('s'):
                    self.save_screenshot(frame)

                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence Threshold: {self.confidence_threshold:.2f}")

                elif key == ord('-'):
                    self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                    print(f"Confidence Threshold: {self.confidence_threshold:.2f}")

        except KeyboardInterrupt:
            print("\nCancelled by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans up and closes all resources"""
        print("Cleaning up...")

        if self.threaded_cam is not None:
            self.threaded_cam.stop()

        if self.ep_camera is not None:
            try:
                self.ep_camera.stop_video_stream()
            except:
                pass

        if self.ep_robot is not None:
            try:
                self.ep_robot.close()
            except:
                pass

        cv2.destroyAllWindows()
        print("Test completed.")


def main():
    """Main function"""
    print("=" * 50)
    print("RoboMaster EP - YOLOv8s Object Detection")
    print("=" * 50)
    print("\nMake sure that:")
    print("  1. RoboMaster EP is turned on")
    print("  2. Connected to the same WiFi network")
    print()

    input("Press ENTER to start...")

    test = YOLOv8sRobotCameraTest(confidence_threshold=0.5)
    test.run()


if __name__ == "__main__":
    main()
