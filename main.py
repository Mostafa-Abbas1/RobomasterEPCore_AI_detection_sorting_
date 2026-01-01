"""
RoboMaster EP Core - AI Object Detection and Sorting
Main entry point for floor-based object sorting
"""

import sys
import logging
import time
import cv2
import numpy as np
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from robomaster import robot, camera as rm_camera
from config import settings
from src.robot_control import ThreadedCamera, RobotMovement, MovementStack
from src.vision import ObjectDetector
from src.sorting import SortingController, ClassBasedStrategy


# ========================================
# CONFIGURATION
# ========================================
BRIGHTNESS_FACTOR = 1.5    # Image brightness for low light (1.5 = 50% brighter)
SCAN_FRAMES = 4              # Multiple frames for reliable detection
BBOX_AREA_THRESHOLD = 50000  # When close enough to grab (higher = closer)
STEP_SIZE = 0.15             # Per-iteration approach distance (meters)
MAX_ITERATIONS = 10          # Maximum approach iterations
ROTATION_STEP = 45           # Degrees to rotate when searching (smaller = more careful)
MAX_ROTATIONS_WITHOUT_FIND = 8  # Full 360
FINAL_APPROACH_DISTANCE = 0.35   # Final distance before grab (meters)


class LiveDisplay:
    """Display live camera feed in separate window"""

    def __init__(self, threaded_camera):
        self.threaded_camera = threaded_camera
        self.stopped = False
        self.current_status = "Initializing..."
        self.current_detections = []
        self.lock = threading.Lock()

    def start(self):
        """Start display thread"""
        threading.Thread(target=self._display_loop, daemon=True).start()
        return self

    def update_status(self, status):
        """Update status text"""
        with self.lock:
            self.current_status = status

    def update_detections(self, detections):
        """Update detections to display"""
        with self.lock:
            self.current_detections = detections.copy() if detections else []

    def _display_loop(self):
        """Main display loop running in thread"""
        while not self.stopped:
            frame = self.threaded_camera.read()

            if frame is not None:
                display_frame = frame.copy()

                # Draw detections
                with self.lock:
                    for det in self.current_detections:
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det.class_name}: {det.confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw status
                    cv2.putText(display_frame, self.current_status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Robot Camera - Live View", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True

            time.sleep(0.03)  # ~30 FPS

    def stop(self):
        """Stop display"""
        self.stopped = True
        cv2.destroyAllWindows()


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def brighten_image(image, factor=1.3):
    """Brighten image for better detection in low light"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def calculate_object_position(detection, frame_width, frame_height):
    """
    Calculate approximate distance and angle to object based on position in frame

    Returns:
        tuple: (distance_estimate, angle_offset)
    """
    center_x, center_y = detection.center
    frame_center_x = frame_width / 2
    offset_x = center_x - frame_center_x

    # RoboMaster SDK: positive z = counter-clockwise (LEFT)
    # If object is RIGHT of center (+offset_x), robot needs to turn RIGHT (negative z)
    angle_offset = -(offset_x / frame_center_x) * 30

    # Estimate distance based on object size
    bbox_width = detection.bbox[2] - detection.bbox[0]
    bbox_height = detection.bbox[3] - detection.bbox[1]
    bbox_area = bbox_width * bbox_height

    return bbox_area, angle_offset


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def print_step(step_num, text):
    """Print step information"""
    print(f"\n>>> Step {step_num}: {text}")
    print("-" * 60)


def main():
    """Main application loop for floor-based sorting"""
    logger = setup_logging()

    print_header("RoboMaster EP Core - Floor-based Object Sorting")

    print("Setup Instructions:")
    print("  - Robot is on the FLOOR")
    print("  - Place bottles and cups around robot on floor")
    print("  - Mark 2 sorting zones on floor:")
    print("    - LEFT zone (90 degrees): For Cups")
    print("    - RIGHT zone (-90 degrees): For Bottles")
    print(f"\nTarget Objects: {settings.OBJECT_CLASSES}")
    print(f"\nZone Configuration:")
    for zone_name, zone_config in settings.SORTING_ZONES.items():
        angle = zone_config.get('navigation_angle', 0)
        direction = "LEFT" if angle > 0 else "RIGHT" if angle < 0 else "STRAIGHT"
        print(f"  - {zone_name}: {direction} ({angle} degrees) - {zone_config.get('description', '')}")

    input("\nPress ENTER when ready to start...")

    # ========================================
    # INITIALIZATION
    # ========================================
    print_step(1, "Initializing Robot")

    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="sta")
        logger.info("Robot connected!")
        print("Robot connected!")

        version = ep_robot.get_version()
        print(f"Robot version: {version}")

    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}")
        print(f"Failed to connect to robot: {e}")
        return 1

    # Get robot components
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_gripper = ep_robot.gripper
    ep_arm = ep_robot.robotic_arm

    # ========================================
    # SETUP SORTING CONTROLLER
    # ========================================
    print_step(2, "Setting up Sorting Controller")

    # Create sorting controller with strategy
    strategy = ClassBasedStrategy(settings.CLASS_ZONE_MAPPING)
    sorting_controller = SortingController(ep_robot, strategy)

    # Setup zones from configuration
    sorting_controller.setup_zones_from_config(settings.SORTING_ZONES)
    print("Sorting zones configured:")
    for zone in sorting_controller.zone_manager.get_all_zones():
        print(f"  - {zone}")

    # Reset position tracker
    RobotMovement.reset_position_tracker()
    print("Position tracker initialized at center")

    # ========================================
    # CAMERA SETUP
    # ========================================
    print_step(3, "Starting Camera System")

    try:
        ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_720P)
        time.sleep(2)

        threaded_cam = ThreadedCamera(ep_camera).start()
        time.sleep(1)

        # Start live display
        live_display = LiveDisplay(threaded_cam).start()
        time.sleep(0.5)

        print("Camera system ready (threaded mode)")
        print("Live camera window opened!")

    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        ep_robot.close()
        return 1

    # ========================================
    # YOLO MODEL LOADING
    # ========================================
    print_step(4, "Loading YOLO Detection Model")

    try:
        detector = ObjectDetector(confidence_threshold=0.25)  # Lower = more detections
        if not detector.load_model():
            raise RuntimeError("Failed to load YOLO model")

        print("YOLO model loaded (confidence threshold: 0.25, model: yolov8s)")

    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        live_display.stop()
        threaded_cam.stop()
        ep_camera.stop_video_stream()
        ep_robot.close()
        return 1

    # ========================================
    # ROBOT ARM POSITION
    # ========================================
    print_step(5, "Preparing Robot Arm for Floor Pickup")

    try:
        print("Lowering arm to floor level...")
        live_display.update_status("Positioning arm for floor pickup...")
        ep_arm.move(x=120, y=-100).wait_for_completed(timeout=5)
        time.sleep(1)
        print("Arm positioned at floor level")

        # Open gripper for first pickup
        sorting_controller.prepare_for_pickup()

    except Exception as e:
        logger.warning(f"Arm positioning: {e}")

    # ========================================
    # MAIN SORTING LOOP
    # ========================================
    print_step(6, "Starting Continuous Scan-and-Sort")

    live_display.update_status("Ready - Press ENTER to start")
    input("\nPress ENTER to start sorting...")

    print("\nStarting continuous scan-and-sort mode!")
    rotations_without_find = 0
    total_objects_processed = 0

    try:
        while rotations_without_find < MAX_ROTATIONS_WITHOUT_FIND:
            print(f"\n{'='*50}")
            print(f"Scanning current view...")
            print(f"{'='*50}")
            live_display.update_status("Scanning for objects...")

            time.sleep(2.0)  # Wait for camera to stabilize

            # Scan multiple frames
            target_detections = []
            frame = None

            for scan_attempt in range(SCAN_FRAMES):
                frame = threaded_cam.read()
                if frame is None:
                    time.sleep(0.3)
                    continue

                bright_frame = brighten_image(frame, factor=BRIGHTNESS_FACTOR)
                detections = detector.detect_objects(bright_frame)
                found_targets = [d for d in detections if d.class_name in settings.OBJECT_CLASSES]

                if found_targets:
                    target_detections = found_targets
                    print(f"  Found object on scan attempt {scan_attempt + 1}/{SCAN_FRAMES}")
                    break

                time.sleep(0.3)

            if frame is None:
                print("No frame available")
                rotations_without_find += 1
                try:
                    ep_chassis.move(x=0, y=0, z=ROTATION_STEP, z_speed=45).wait_for_completed(timeout=5)
                except Exception as e:
                    logger.warning(f"Rotation failed: {e}")
                continue

            live_display.update_detections(target_detections)

            if not target_detections:
                print("No objects in current view, rotating to search...")
                rotations_without_find += 1
                try:
                    ep_chassis.move(x=0, y=0, z=ROTATION_STEP, z_speed=45).wait_for_completed(timeout=5)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Rotation failed: {e}")
                continue

            # Found object - reset counter
            rotations_without_find = 0
            total_objects_processed += 1

            obj = target_detections[0]
            print(f"\nFound {obj.class_name} (confidence: {obj.confidence:.2f})")
            live_display.update_status(f"Found {obj.class_name} - Processing...")

            try:
                # ========================================
                # ITERATIVE APPROACH TO OBJECT
                # ========================================
                print("\nStarting iterative approach to object...")
                live_display.update_status(f"Approaching {obj.class_name}...")

                h, w = frame.shape[:2]
                bbox_area, angle_offset = calculate_object_position(obj, w, h)
                total_distance_traveled = 0
                iteration = 0
                
                # Create movement stack to record all movements for precise return
                movement_stack = MovementStack()

                print(f"Initial bbox_area: {bbox_area:.0f} (target: {BBOX_AREA_THRESHOLD})")

                while bbox_area < BBOX_AREA_THRESHOLD and iteration < MAX_ITERATIONS:
                    iteration += 1
                    print(f"\n  [Iteration {iteration}] bbox_area={bbox_area:.0f}")

                    # Center on object
                    if abs(angle_offset) > 2:
                        print(f"  Centering ({angle_offset:.1f} degrees)...")
                        try:
                            ep_chassis.move(x=0, y=0, z=angle_offset, z_speed=30).wait_for_completed(timeout=3)
                            movement_stack.record(0, 0, angle_offset)  # Record rotation
                            time.sleep(0.3)
                        except Exception as e:
                            logger.warning(f"Centering failed: {e}")

                    # Move forward
                    print(f"  Moving {STEP_SIZE}m forward...")
                    try:
                        ep_chassis.move(x=STEP_SIZE, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=5)
                        movement_stack.record(STEP_SIZE, 0, 0)  # Record forward movement
                        total_distance_traveled += STEP_SIZE
                        time.sleep(0.4)
                    except Exception as e:
                        logger.warning(f"Movement failed: {e}")
                        break

                    # Re-scan
                    time.sleep(0.3)
                    new_frame = threaded_cam.read()
                    if new_frame is None:
                        print("  No frame, stopping approach")
                        break

                    bright_new_frame = brighten_image(new_frame, factor=BRIGHTNESS_FACTOR)
                    new_detections = detector.detect_objects(bright_new_frame)
                    target_det = [d for d in new_detections if d.class_name == obj.class_name]

                    if not target_det:
                        print("  Object lost! Stopping approach")
                        break

                    current_obj = target_det[0]
                    h, w = new_frame.shape[:2]
                    bbox_area, angle_offset = calculate_object_position(current_obj, w, h)
                    live_display.update_detections(target_det)

                    print(f"  New bbox_area: {bbox_area:.0f}, angle: {angle_offset:.1f} degrees")

                if bbox_area >= BBOX_AREA_THRESHOLD:
                    print(f"\nClose enough! bbox_area={bbox_area:.0f}")
                else:
                    print(f"\nStopped after {iteration} iterations")

                print(f"Total distance traveled: {total_distance_traveled:.2f}m")

                # Final centering
                print("\nFinal centering before grab...")
                time.sleep(0.3)
                final_frame = threaded_cam.read()
                if final_frame is not None:
                    bright_final = brighten_image(final_frame, factor=BRIGHTNESS_FACTOR)
                    final_det = detector.detect_objects(bright_final)
                    final_target = [d for d in final_det if d.class_name == obj.class_name]

                    if final_target:
                        h, w = final_frame.shape[:2]
                        _, final_angle = calculate_object_position(final_target[0], w, h)
                        if abs(final_angle) > 2:
                            print(f"Final adjustment ({final_angle:.1f} degrees)...")
                            try:
                                ep_chassis.move(x=0, y=0, z=final_angle, z_speed=20).wait_for_completed(timeout=3)
                                movement_stack.record(0, 0, final_angle)  # Record final rotation
                            except:
                                pass

                # Final approach
                print(f"Final approach ({FINAL_APPROACH_DISTANCE}m)...")
                try:
                    ep_chassis.move(x=FINAL_APPROACH_DISTANCE, y=0, z=0, xy_speed=0.15).wait_for_completed(timeout=5)
                    movement_stack.record(FINAL_APPROACH_DISTANCE, 0, 0)  # Record final approach
                    total_distance_traveled += FINAL_APPROACH_DISTANCE
                except Exception as e:
                    logger.warning(f"Final approach failed: {e}")

                # ========================================
                # SORT THE OBJECT
                # ========================================
                live_display.update_status(f"Sorting {obj.class_name}...")
                live_display.update_detections([])

                if sorting_controller.sort_object(obj, approach_distance=total_distance_traveled, movement_stack=movement_stack):
                    print(f"\nObject sorted successfully!")
                    stats = sorting_controller.get_sorting_statistics()
                    print(f"Total sorted: {stats['total_sorted']}")
                else:
                    print(f"\nFailed to sort object")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user!")
                break
            except Exception as e:
                logger.error(f"Error during sorting: {e}")
                # Recovery
                try:
                    ep_gripper.open(power=50)
                    time.sleep(0.5)
                    ep_gripper.pause()
                    ep_chassis.move(x=-0.2, y=0, z=0, xy_speed=0.3).wait_for_completed()
                except:
                    pass

            print("Continuing to next scan...\n")
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # ========================================
    # RESULTS
    # ========================================
    print_header("Sorting Complete!")

    stats = sorting_controller.get_sorting_statistics()
    print(f"Total objects processed: {total_objects_processed}")
    print(f"Successfully sorted:     {stats['total_sorted']}")
    print(f"Failed:                  {stats['total_failed']}")
    if total_objects_processed > 0:
        print(f"Success rate:            {stats['success_rate']:.1f}%")

    print("\nZone summary:")
    for zone in sorting_controller.zone_manager.get_all_zones():
        info = zone.get_info()
        print(f"  - {zone.name}: {info['current_count']}/{info['capacity']} objects")

    # ========================================
    # CLEANUP
    # ========================================
    print("\nCleaning up...")
    live_display.update_status("Demo Complete!")

    try:
        ep_arm.move(x=0, y=0).wait_for_completed()
    except:
        pass

    time.sleep(2)

    live_display.stop()
    threaded_cam.stop()
    ep_camera.stop_video_stream()
    ep_robot.close()

    print("\nDemo complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
