"""
Floor-based Sorting Demo
RoboMaster EP Core operates on the floor, scans 360°, finds objects and sorts them

Setup:
- Robot on the floor
- Objects (bottles & cups) placed around the robot on the floor
- Define target zones on the floor (mark with tape/paper)
- No table needed!
"""

import sys
import time
import cv2
import numpy as np
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ========================================
# CONFIGURATION
# ========================================
BRIGHTNESS_FACTOR = 1.8  # Image brightness for low light (1.0 = no change, higher = brighter)

from robomaster import robot, camera as rm_camera
from config import settings
from src.vision.detection import ObjectDetector
from src.robot_control import ThreadedCamera


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


def brighten_image(image, factor=1.3):
    """
    Brighten image for better detection in low light

    Args:
        image: Input image (BGR)
        factor: Brightness factor (1.0 = no change, >1.0 = brighter)

    Returns:
        Brightened image
    """
    # Convert to HSV and increase V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase brightness
    v = np.clip(v * factor, 0, 255).astype(np.uint8)

    # Merge and convert back
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def print_step(step_num, text):
    """Print step information"""
    print(f"\n>>> Step {step_num}: {text}")
    print("-" * 60)


def calculate_object_position(detection, frame_width, frame_height):
    """
    Calculate approximate distance and angle to object based on position in frame

    Args:
        detection: DetectedObject
        frame_width: Width of camera frame
        frame_height: Height of camera frame

    Returns:
        tuple: (distance_estimate, angle_offset)
    """
    # Get center of object
    center_x, center_y = detection.center

    # Calculate horizontal offset from center (for rotation)
    frame_center_x = frame_width / 2
    offset_x = center_x - frame_center_x

    # Approximate angle offset (very rough estimation)
    # Assuming ~60° horizontal FOV
    # RoboMaster SDK: positive z = counter-clockwise (left), negative z = clockwise (right)
    # If object is RIGHT of center (+offset_x), robot needs to turn RIGHT (negative z)
    # If object is LEFT of center (-offset_x), robot needs to turn LEFT (positive z)
    angle_offset = -(offset_x / frame_center_x) * 30  # degrees (negated for correct direction)

    # Debug output
    print(f"  [Debug] center_x={center_x:.0f}, frame_center={frame_center_x:.0f}, offset_x={offset_x:.0f}°")

    # Estimate distance based on object size
    # Larger bbox = closer object
    bbox_width = detection.bbox[2] - detection.bbox[0]
    bbox_height = detection.bbox[3] - detection.bbox[1]
    bbox_area = bbox_width * bbox_height

    # Distance estimation - INCREASED by ~15cm to reach objects properly
    # Tune these values based on your objects!
    if bbox_area > 50000:  # Very close
        distance = 0.45  # was 0.3, added 15cm
    elif bbox_area > 20000:  # Medium
        distance = 0.65  # was 0.5, added 15cm
    elif bbox_area > 10000:  # Far
        distance = 0.85  # was 0.7, added 15cm
    else:  # Very far
        distance = 1.15  # was 1.0, added 15cm

    print(f"  [Debug] bbox_area={bbox_area}, distance={distance}m")

    return distance, angle_offset


def main():
    print_header("RoboMaster EP Core - Floor-based Object Sorting")

    print("Setup Instructions:")
    print("  ✓ Robot is on the FLOOR")
    print("  ✓ Place bottles and cups around robot on floor")
    print("  ✓ Mark 2 sorting zones on floor:")
    print("    - LEFT zone: For Cups")
    print("    - RIGHT zone: For Bottles")
    print(f"\nTarget Objects: {settings.OBJECT_CLASSES}")

    input("\nPress ENTER when ready to start...")

    # ========================================
    # INITIALIZATION
    # ========================================
    print_step(1, "Initializing Robot")

    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="sta")
        print("✓ Robot connected!")

        version = ep_robot.get_version()
        print(f"✓ Robot version: {version}")

    except Exception as e:
        print(f"✗ Failed to connect to robot: {e}")
        return 1

    # Get robot components
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_gripper = ep_robot.gripper
    ep_arm = ep_robot.robotic_arm

    # ========================================
    # CAMERA SETUP
    # ========================================
    print_step(2, "Starting Camera System")

    try:
        ep_camera.start_video_stream(display=False, resolution=rm_camera.STREAM_720P)
        time.sleep(2)

        threaded_cam = ThreadedCamera(ep_camera).start()
        time.sleep(1)

        # Start live display
        live_display = LiveDisplay(threaded_cam).start()
        time.sleep(0.5)

        print("✓ Camera system ready (threaded mode)")
        print("✓ Live camera window opened!")

    except Exception as e:
        print(f"✗ Failed to start camera: {e}")
        ep_robot.close()
        return 1

    # ========================================
    # YOLO MODEL LOADING
    # ========================================
    print_step(3, "Loading YOLO Detection Model")

    try:
        detector = ObjectDetector(confidence_threshold=0.35)  # Lowered for dark environments
        if not detector.load_model():
            raise RuntimeError("Failed to load YOLO model")

        print(f"✓ YOLO model loaded (confidence threshold: 0.35, model: yolov8s)")

    except Exception as e:
        print(f"✗ Failed to load YOLO: {e}")
        live_display.stop()
        threaded_cam.stop()
        ep_camera.stop_video_stream()
        ep_robot.close()
        return 1

    # ========================================
    # READY TO START
    # ========================================
    print_step(4, "Ready to Start Sorting")

    print("Robot will continuously scan and grab objects as it finds them")
    print("Press ENTER to start sorting...")
    live_display.update_status("Ready to start sorting")

    input()

    print("\n✓ Starting continuous scan-and-grab mode!")

    # ========================================
    # ROBOT ARM POSITION - LOWER FOR FLOOR
    # ========================================
    print_step(5, "Preparing Robot Arm for Floor Pickup")

    try:
        # Lower arm to LOWEST position for floor-level objects
        print("Lowering arm to floor level (lowest position)...")
        live_display.update_status("Positioning arm for floor pickup...")

        # Move arm forward and DOWN to floor level
        # x=120: Forward position to reach objects
        # y=-100: LOWEST position for floor pickup (cups & bottles)
        ep_arm.move(x=120, y=-100).wait_for_completed(timeout=5)
        time.sleep(1)
        print("✓ Arm positioned at LOWEST level for floor pickup")
    except Exception as e:
        print(f"⚠ Arm positioning: {e}")

    # ========================================
    # CONTINUOUS SCAN-AND-GRAB PROCESS
    # ========================================
    print_step(6, "Starting Continuous Scan-and-Grab")

    sorted_count = 0
    failed_count = 0

    ROTATION_STEP = 45  # Rotate 45° when searching
    MAX_ROTATIONS_WITHOUT_FIND = 8  # Full 360° = 8 steps of 45°

    rotations_without_find = 0
    total_objects_processed = 0

    while rotations_without_find < MAX_ROTATIONS_WITHOUT_FIND:
        print(f"\n{'='*50}")
        print(f"Scanning current view...")
        print(f"{'='*50}")
        live_display.update_status("Scanning for objects...")

        # Wait for camera to stabilize (increased for better detection)
        time.sleep(1.2)

        # Scan multiple frames to improve detection reliability
        SCAN_FRAMES = 3
        target_detections = []
        frame = None

        for scan_attempt in range(SCAN_FRAMES):
            frame = threaded_cam.read()

            if frame is None:
                time.sleep(0.3)
                continue

            # Brighten image for better detection in low light
            bright_frame = brighten_image(frame, factor=BRIGHTNESS_FACTOR)

            # Detect objects in brightened image
            detections = detector.detect_objects(bright_frame)
            found_targets = [d for d in detections if d.class_name in settings.OBJECT_CLASSES]

            if found_targets:
                target_detections = found_targets
                print(f"  ✓ Found object on scan attempt {scan_attempt + 1}/{SCAN_FRAMES}")
                break

            time.sleep(0.3)  # Wait before next scan attempt

        if frame is None:
            print("⚠ No frame available after multiple attempts")
            rotations_without_find += 1
            # Rotate and continue
            try:
                ep_chassis.move(x=0, y=0, z=ROTATION_STEP, z_speed=45).wait_for_completed(timeout=5)
                time.sleep(0.5)
            except Exception as e:
                print(f"✗ Rotation failed/timeout: {e}")
            continue

        # Update live display
        live_display.update_detections(target_detections)

        if not target_detections:
            # No objects found in current view
            print("→ No objects in current view, rotating to search...")
            rotations_without_find += 1

            # Rotate to next position
            try:
                ep_chassis.move(x=0, y=0, z=ROTATION_STEP, z_speed=45).wait_for_completed(timeout=5)
                time.sleep(0.5)
            except Exception as e:
                print(f"✗ Rotation failed/timeout: {e}")
            continue

        # Found object(s)! Reset counter
        rotations_without_find = 0
        total_objects_processed += 1

        # Take the first detected object
        obj = target_detections[0]

        print(f"\n✓ Found {obj.class_name} (confidence: {obj.confidence:.2f})")
        live_display.update_status(f"Found {obj.class_name} - Processing...")

        try:
            # Get target zone
            zone_name = settings.CLASS_ZONE_MAPPING.get(obj.class_name, "zone_default")
            print(f"Target zone: {zone_name}")

            # ============================================
            # OPEN GRIPPER IMMEDIATELY
            # ============================================
            print("\n→ Opening gripper fully...")
            ep_gripper.open(power=100)
            time.sleep(1.5)
            ep_gripper.pause()
            print("✓ Gripper fully open")

            # ============================================
            # ITERATIVE VISUAL SERVOING APPROACH
            # ============================================
            print("\n→ Starting iterative approach to object...")
            live_display.update_status(f"Approaching {obj.class_name} iteratively...")

            BBOX_AREA_THRESHOLD = 25000  # When bbox is this big, we're close enough (reduced to stop earlier)
            STEP_SIZE = 0.15  # Move 15cm per iteration
            MAX_ITERATIONS = 10
            total_distance_traveled = 0

            iteration = 0
            current_bbox_area = 0

            # Get initial measurements
            h, w = frame.shape[:2]
            _, angle_offset = calculate_object_position(obj, w, h)
            bbox_width = obj.bbox[2] - obj.bbox[0]
            bbox_height = obj.bbox[3] - obj.bbox[1]
            current_bbox_area = bbox_width * bbox_height

            print(f"→ Initial bbox_area: {current_bbox_area:.0f} (target: {BBOX_AREA_THRESHOLD})")

            # ITERATIVE LOOP: Approach step-by-step
            while current_bbox_area < BBOX_AREA_THRESHOLD and iteration < MAX_ITERATIONS:
                iteration += 1
                print(f"\n  [Iteration {iteration}] bbox_area={current_bbox_area:.0f}")

                # STEP 1: Center the object
                if abs(angle_offset) > 2:
                    print(f"  → Centering ({angle_offset:.1f}°)...")
                    try:
                        ep_chassis.move(x=0, y=0, z=angle_offset, z_speed=30).wait_for_completed(timeout=3)
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"  ✗ Centering failed/timeout: {e}")

                # STEP 2: Move forward one step
                print(f"  → Moving {STEP_SIZE}m forward...")
                try:
                    ep_chassis.move(x=STEP_SIZE, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=5)
                    total_distance_traveled += STEP_SIZE
                    time.sleep(0.4)
                except Exception as e:
                    print(f"  ✗ Movement failed/timeout: {e}")
                    break

                # STEP 3: Re-scan and measure
                time.sleep(0.3)
                new_frame = threaded_cam.read()
                if new_frame is None:
                    print("  ⚠ No frame, stopping approach")
                    break

                bright_new_frame = brighten_image(new_frame, factor=BRIGHTNESS_FACTOR)
                new_detections = detector.detect_objects(bright_new_frame)
                target_det = [d for d in new_detections if d.class_name == obj.class_name]

                if not target_det:
                    print("  ⚠ Object lost! Stopping approach")
                    break

                # Update measurements
                current_obj = target_det[0]
                h, w = new_frame.shape[:2]
                _, angle_offset = calculate_object_position(current_obj, w, h)

                bbox_width = current_obj.bbox[2] - current_obj.bbox[0]
                bbox_height = current_obj.bbox[3] - current_obj.bbox[1]
                current_bbox_area = bbox_width * bbox_height

                # Update display
                live_display.update_detections(target_det)

                print(f"  → New bbox_area: {current_bbox_area:.0f}, angle: {angle_offset:.1f}°")

            # Check if we got close enough
            if current_bbox_area >= BBOX_AREA_THRESHOLD:
                print(f"\n✓ Close enough! bbox_area={current_bbox_area:.0f}")
            else:
                print(f"\n⚠ Stopped after {iteration} iterations, bbox_area={current_bbox_area:.0f}")

            print(f"→ Total distance traveled: {total_distance_traveled:.2f}m")

            # FINAL CENTERING
            print("\n→ Final centering before grab...")
            time.sleep(0.3)
            final_frame = threaded_cam.read()
            if final_frame is not None:
                bright_final_frame = brighten_image(final_frame, factor=BRIGHTNESS_FACTOR)
                final_detections = detector.detect_objects(bright_final_frame)
                final_target = [d for d in final_detections if d.class_name == obj.class_name]

                if final_target:
                    h, w = final_frame.shape[:2]
                    _, final_angle = calculate_object_position(final_target[0], w, h)

                    if abs(final_angle) > 2:
                        print(f"→ Final adjustment ({final_angle:.1f}°)...")
                        try:
                            ep_chassis.move(x=0, y=0, z=final_angle, z_speed=20).wait_for_completed(timeout=3)
                            time.sleep(0.3)
                        except Exception as e:
                            print(f"✗ Final centering failed/timeout: {e}")

            # STEP: Final small approach (gripper already open)
            print(f"→ Final approach (0.20m with open gripper)...")
            try:
                ep_chassis.move(x=0.20, y=0, z=0, xy_speed=0.15).wait_for_completed(timeout=5)
                total_distance_traveled += 0.20
                time.sleep(0.5)
            except Exception as e:
                print(f"✗ Final approach failed/timeout: {e}")

            # STEP: Close gripper to grab
            print("→ Grabbing object (closing gripper)...")
            live_display.update_status(f"Grabbing {obj.class_name}...")
            ep_gripper.close(power=50)
            time.sleep(1.5)
            ep_gripper.pause()

            print("✓ Object grabbed!")

            # STEP: Lift object slightly
            print("→ Lifting object...")
            try:
                ep_arm.move(x=0, y=50).wait_for_completed(timeout=3)
                time.sleep(0.5)
                print("✓ Object lifted")
            except Exception as e:
                print(f"⚠ Lift failed/timeout: {e}")

            # STEP: Move back to starting position
            print(f"→ Returning to center ({total_distance_traveled:.2f}m back)...")
            try:
                ep_chassis.move(x=-total_distance_traveled, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=10)
                time.sleep(0.5)
            except Exception as e:
                print(f"✗ Return to center failed/timeout: {e}")

            # STEP 6: Navigate to zone (simplified)
            print(f"→ Moving to {zone_name}...")
            live_display.update_status(f"Moving to {zone_name}...")
            live_display.update_detections([])  # Clear detections during transport

            try:
                # Simplified zone navigation
                # Zone A (cups) = Left side
                # Zone B (bottles) = Right side
                if zone_name == "zone_a":
                    # Turn left, drive forward, place
                    print("→ Turning left to Cup zone...")
                    ep_chassis.move(x=0, y=0, z=-90, z_speed=45).wait_for_completed(timeout=5)
                    time.sleep(0.5)
                    print("→ Driving to zone...")
                    ep_chassis.move(x=0.8, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=10)
                    time.sleep(0.5)

                elif zone_name == "zone_b":
                    # Turn right, drive forward, place
                    print("→ Turning right to Bottle zone...")
                    ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed(timeout=5)
                    time.sleep(0.5)
                    print("→ Driving to zone...")
                    ep_chassis.move(x=0.8, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=10)
                    time.sleep(0.5)

            except Exception as e:
                print(f"✗ Zone navigation failed/timeout: {e}")

            # STEP 7: Release object
            print("→ Releasing object...")
            ep_gripper.open(power=50)
            time.sleep(1)
            ep_gripper.pause()

            print("✓ Object placed!")

            # STEP: Lower arm back to floor level and keep gripper open
            print("→ Lowering arm back to floor level...")
            try:
                ep_arm.move(x=0, y=-50).wait_for_completed(timeout=3)
                time.sleep(0.3)
                ep_gripper.open(power=100)
                time.sleep(1)
                ep_gripper.pause()
                print("✓ Arm lowered, gripper open")
            except Exception as e:
                print(f"⚠ Arm lowering failed/timeout: {e}")

            # STEP 8: Return to center (simplified)
            print("→ Returning to center...")
            try:
                # Drive back
                print("→ Driving back...")
                ep_chassis.move(x=-0.8, y=0, z=0, xy_speed=0.3).wait_for_completed(timeout=10)
                time.sleep(0.5)

                # Turn back to forward
                if zone_name == "zone_a":
                    print("→ Turning right to face forward...")
                    ep_chassis.move(x=0, y=0, z=90, z_speed=45).wait_for_completed(timeout=5)
                elif zone_name == "zone_b":
                    print("→ Turning left to face forward...")
                    ep_chassis.move(x=0, y=0, z=-90, z_speed=45).wait_for_completed(timeout=5)

                time.sleep(0.5)
                print("✓ Back at center position")

            except Exception as e:
                print(f"✗ Return to center failed/timeout: {e}")

            sorted_count += 1
            print(f"✓ Object sorted successfully! (Total: {sorted_count})")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user!")
            break
        except Exception as e:
            print(f"✗ Failed to sort object: {e}")
            failed_count += 1
            # Try to recover - open gripper and back up
            print("→ Attempting recovery...")
            try:
                ep_gripper.open(power=50)
                time.sleep(0.5)
                ep_gripper.pause()
                # Try to back up a bit
                ep_chassis.move(x=-0.2, y=0, z=0, xy_speed=0.3).wait_for_completed()
            except:
                pass

        # Small pause before next scan
        print("→ Continuing to next scan...\n")
        time.sleep(1.5)

    # End of main loop
    print(f"\n✓ Scan complete! Processed {total_objects_processed} objects in total.")

    # ========================================
    # RESULTS
    # ========================================
    print_header("Sorting Complete!")

    print(f"Total objects processed: {total_objects_processed}")
    print(f"Successfully sorted:     {sorted_count}")
    print(f"Failed:                  {failed_count}")
    if total_objects_processed > 0:
        print(f"Success rate:            {sorted_count/total_objects_processed*100:.1f}%")

    # ========================================
    # CLEANUP
    # ========================================
    print("\nCleaning up...")
    live_display.update_status("Cleanup - Demo Complete!")

    # Return arm to neutral
    try:
        ep_arm.move(x=0, y=0).wait_for_completed()
    except:
        pass

    time.sleep(2)  # Show final status briefly

    live_display.stop()
    threaded_cam.stop()
    ep_camera.stop_video_stream()
    ep_robot.close()

    print("\n✓ Demo complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
