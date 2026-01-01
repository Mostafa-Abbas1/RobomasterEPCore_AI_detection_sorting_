"""
Robot Scanner Module
360-degree scanning for object detection
"""

import time
import logging
from typing import List
from ..vision.detection import DetectedObject

logger = logging.getLogger(__name__)


class RobotScanner:
    """
    Handles 360-degree scanning and object detection
    """

    def __init__(self, robot, camera, detector):
        """
        Initialize scanner

        Args:
            robot: Connected robot instance
            camera: Camera instance (can be ThreadedCamera)
            detector: ObjectDetector instance
        """
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.chassis = robot.chassis if robot else None

    def scan_360(self, steps: int = 8, target_classes: List[str] = None) -> List[DetectedObject]:
        """
        Perform 360-degree scan to detect objects

        Args:
            steps: Number of scanning positions (default 8 = 45° increments)
            target_classes: Optional list of classes to filter for

        Returns:
            List[DetectedObject]: All detected objects from the scan
        """
        if not self.chassis:
            logger.error("Chassis not available - cannot scan")
            return []

        logger.info(f"Starting 360° scan with {steps} positions...")

        all_detections = []
        angle_per_step = 360.0 / steps
        seen_objects = set()  # Track unique objects by position

        for i in range(steps):
            logger.info(f"Scan position {i+1}/{steps} ({i*angle_per_step:.0f}°)")

            # Wait for camera to stabilize
            time.sleep(0.5)

            # Get frame
            frame = None
            if hasattr(self.camera, 'read'):
                # ThreadedCamera
                frame = self.camera.read()
            else:
                # Regular camera
                frame = self.camera.get_frame()

            if frame is None:
                logger.warning(f"No frame at position {i+1}")
                continue

            # Detect objects
            detections = self.detector.detect_objects(frame)

            # Filter by target classes if specified
            if target_classes:
                detections = [d for d in detections if d.class_name in target_classes]

            # Add unique detections
            for det in detections:
                # Simple uniqueness check based on class and approximate position
                obj_key = f"{det.class_name}_{det.center[0]//50}_{det.center[1]//50}"

                if obj_key not in seen_objects:
                    seen_objects.add(obj_key)
                    all_detections.append(det)
                    logger.info(f"  Found: {det.class_name} (conf: {det.confidence:.2f})")

            # Rotate to next position (except on last step)
            if i < steps - 1:
                logger.debug(f"Rotating {angle_per_step:.0f}° to next position...")
                try:
                    self.chassis.move(x=0, y=0, z=angle_per_step, z_speed=45).wait_for_completed()
                except Exception as e:
                    logger.error(f"Failed to rotate: {e}")

        logger.info(f"Scan complete! Found {len(all_detections)} unique objects")
        return all_detections

    def scan_area(self, angle_range: float = 180.0, steps: int = 4,
                  target_classes: List[str] = None) -> List[DetectedObject]:
        """
        Scan a specific angle range (e.g., front 180 degrees)

        Args:
            angle_range: Total angle to scan (default 180°)
            steps: Number of positions
            target_classes: Optional list of classes to filter for

        Returns:
            List[DetectedObject]: Detected objects
        """
        if not self.chassis:
            logger.error("Chassis not available - cannot scan")
            return []

        logger.info(f"Scanning {angle_range}° range with {steps} positions...")

        all_detections = []
        angle_per_step = angle_range / (steps - 1) if steps > 1 else 0
        start_angle = -angle_range / 2  # Start from left

        # Rotate to start position
        logger.info(f"Rotating to start position ({start_angle:.0f}°)")
        try:
            self.chassis.move(x=0, y=0, z=start_angle, z_speed=45).wait_for_completed()
        except Exception as e:
            logger.error(f"Failed to rotate to start: {e}")
            return []

        time.sleep(1)

        # Scan
        seen_objects = set()
        for i in range(steps):
            logger.info(f"Scan position {i+1}/{steps}")

            time.sleep(0.5)

            # Get frame
            frame = None
            if hasattr(self.camera, 'read'):
                frame = self.camera.read()
            else:
                frame = self.camera.get_frame()

            if frame is None:
                logger.warning(f"No frame at position {i+1}")
                continue

            # Detect
            detections = self.detector.detect_objects(frame)

            if target_classes:
                detections = [d for d in detections if d.class_name in target_classes]

            for det in detections:
                obj_key = f"{det.class_name}_{det.center[0]//50}_{det.center[1]//50}"
                if obj_key not in seen_objects:
                    seen_objects.add(obj_key)
                    all_detections.append(det)
                    logger.info(f"  Found: {det.class_name} (conf: {det.confidence:.2f})")

            # Rotate to next position
            if i < steps - 1:
                try:
                    self.chassis.move(x=0, y=0, z=angle_per_step, z_speed=45).wait_for_completed()
                except Exception as e:
                    logger.error(f"Failed to rotate: {e}")

        # Return to center
        logger.info("Returning to center position")
        return_angle = -(start_angle + angle_per_step * (steps - 1))
        try:
            self.chassis.move(x=0, y=0, z=return_angle, z_speed=45).wait_for_completed()
        except Exception as e:
            logger.error(f"Failed to return to center: {e}")

        logger.info(f"Area scan complete! Found {len(all_detections)} unique objects")
        return all_detections

    def quick_scan(self, target_classes: List[str] = None) -> List[DetectedObject]:
        """
        Quick scan of current view only (no rotation)

        Args:
            target_classes: Optional list of classes to filter for

        Returns:
            List[DetectedObject]: Detected objects
        """
        logger.info("Quick scan (current view only)...")

        # Get frame
        frame = None
        if hasattr(self.camera, 'read'):
            frame = self.camera.read()
        else:
            frame = self.camera.get_frame()

        if frame is None:
            logger.error("No frame available")
            return []

        # Detect
        detections = self.detector.detect_objects(frame)

        if target_classes:
            detections = [d for d in detections if d.class_name in target_classes]

        logger.info(f"Found {len(detections)} objects in current view")
        for det in detections:
            logger.info(f"  - {det.class_name} (conf: {det.confidence:.2f})")

        return detections
