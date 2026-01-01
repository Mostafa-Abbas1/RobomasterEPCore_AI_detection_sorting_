"""
Sorting Logic Module
Main controller for object sorting operations with correct zone navigation
"""

from typing import List, Optional, Tuple
from ..vision.detection import DetectedObject
from .strategy import SortingStrategy
from .zones import ZoneManager, SortingZone
import logging
import time

logger = logging.getLogger(__name__)


class SortingController:
    """
    Main controller for sorting detected objects.
    Handles the complete sorting workflow including navigation, pickup, and placement.
    """

    def __init__(self, robot, strategy: Optional[SortingStrategy] = None):
        """
        Initialize the sorting controller

        Args:
            robot: Connected robot instance (RoboMaster EP)
            strategy: Sorting strategy to use
        """
        self.robot = robot
        self.strategy = strategy
        self.zone_manager = ZoneManager()
        self.sorted_objects_count = 0
        self.failed_count = 0

        # Get robot components
        self.chassis = robot.chassis if robot else None
        self.gripper = robot.gripper if robot else None
        self.arm = robot.robotic_arm if robot else None

        # Navigation settings
        self.approach_speed = 0.3
        self.rotation_speed = 45
        self.zone_distance = 0.8  # Distance to drive into zone

    def setup_zones_from_config(self, zones_config: dict):
        """
        Setup sorting zones from configuration dictionary

        Args:
            zones_config: Dictionary with zone configurations from settings.py
        """
        for zone_name, zone_config in zones_config.items():
            self.zone_manager.create_zone(
                name=zone_name,
                position=zone_config.get("position", (0.5, 0.0)),
                capacity=zone_config.get("capacity", 10),
                navigation_angle=zone_config.get("navigation_angle", 0),
                return_angle=zone_config.get("return_angle", 0),
                description=zone_config.get("description", "")
            )
            logger.info(f"Created zone: {zone_name} (angle: {zone_config.get('navigation_angle', 0)}°)")

    def set_strategy(self, strategy: SortingStrategy):
        """Set the sorting strategy"""
        self.strategy = strategy
        logger.info(f"Set sorting strategy to {strategy.__class__.__name__}")

    def navigate_to_zone(self, zone: SortingZone) -> bool:
        """
        Navigate robot to a sorting zone using time-based movement.
        Uses drive_speed() for more direct control without position drift.

        Args:
            zone: Target SortingZone

        Returns:
            bool: True if navigation successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available")
                return False

            nav_angle = zone.navigation_angle
            distance = zone.position[0]

            # WICHTIG: Erst komplett stoppen!
            self.chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.5)

            if abs(nav_angle) >= 1:
                # Step 1: Drehen mit Zeit-basierter Kontrolle
                direction = "LEFT" if nav_angle > 0 else "RIGHT"
                logger.info(f"Turning {abs(nav_angle)}° {direction} to face {zone.name}...")
                
                # Berechne Drehzeit: bei 30°/s brauchen wir angle/30 Sekunden
                rotation_speed = 30  # Grad pro Sekunde
                rotation_time = abs(nav_angle) / rotation_speed
                
                # Drehe mit drive_speed (z ist in Grad/Sekunde)
                # WICHTIG: drive_speed z-Richtung ist umgekehrt zu move()!
                # Positiver Winkel (LEFT) braucht negative z-speed
                z_direction = -rotation_speed if nav_angle > 0 else rotation_speed
                self.chassis.drive_speed(x=0, y=0, z=z_direction)
                time.sleep(rotation_time)
                
                # STOPPEN nach Drehung!
                self.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(1.0)  # Warten bis Roboter stabil ist
            
            # Step 2: GERADEAUS fahren mit Zeit-basierter Kontrolle
            logger.info(f"Driving {distance}m STRAIGHT into {zone.name}...")
            
            # Berechne Fahrzeit: bei 0.2 m/s brauchen wir distance/0.2 Sekunden
            drive_speed = 0.2  # Meter pro Sekunde
            drive_time = distance / drive_speed
            
            # Fahre NUR vorwärts (y=0, z=0 für gerade Linie!)
            self.chassis.drive_speed(x=drive_speed, y=0, z=0)
            time.sleep(drive_time)
            
            # STOPPEN!
            self.chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.5)

            logger.info(f"Arrived at zone {zone.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to navigate to zone: {e}")
            return False

    def return_from_zone(self, zone: SortingZone) -> bool:
        """
        Return robot from zone back to center position using time-based movement.

        Args:
            zone: Zone to return from

        Returns:
            bool: True if return successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available")
                return False

            distance = zone.position[0]
            return_angle = zone.return_angle

            # WICHTIG: Erst komplett stoppen!
            self.chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.5)

            # Step 1: GERADE rückwärts fahren
            logger.info(f"Reversing {distance}m STRAIGHT from {zone.name}...")
            
            drive_speed = 0.2
            drive_time = distance / drive_speed
            
            # Fahre NUR rückwärts (y=0, z=0!)
            self.chassis.drive_speed(x=-drive_speed, y=0, z=0)
            time.sleep(drive_time)
            
            # STOPPEN!
            self.chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(1.0)

            # Step 2: Zurückdrehen
            if abs(return_angle) > 1:
                direction = "LEFT" if return_angle > 0 else "RIGHT"
                logger.info(f"Turning {abs(return_angle)}° {direction} to face forward...")
                
                rotation_speed = 30
                rotation_time = abs(return_angle) / rotation_speed
                
                # WICHTIG: drive_speed z-Richtung ist umgekehrt zu move()!
                z_direction = -rotation_speed if return_angle > 0 else rotation_speed
                self.chassis.drive_speed(x=0, y=0, z=z_direction)
                time.sleep(rotation_time)
                
                # STOPPEN!
                self.chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(1.0)

            logger.info("Returned to center position")
            return True

        except Exception as e:
            logger.error(f"Failed to return from zone: {e}")
            return False

    def pick_up_object(self, grip_power: int = 100) -> bool:
        """
        Pick up object at current position using gripper.

        Args:
            grip_power: Gripper closing power (1-100)

        Returns:
            bool: True if pickup successful
        """
        try:
            if not self.gripper:
                logger.error("Gripper not available")
                return False

            logger.info("Closing gripper to grab object...")
            self.gripper.close(power=grip_power)
            time.sleep(1.5)
            self.gripper.pause()

            logger.info("Object grabbed")
            return True

        except Exception as e:
            logger.error(f"Failed to pick up object: {e}")
            return False

    def release_object(self, release_power: int = 50) -> bool:
        """
        Release held object by opening gripper.

        Args:
            release_power: Gripper opening power (1-100)

        Returns:
            bool: True if release successful
        """
        try:
            if not self.gripper:
                logger.error("Gripper not available")
                return False

            logger.info("Opening gripper to release object...")
            self.gripper.open(power=release_power)
            time.sleep(1.0)
            self.gripper.pause()

            logger.info("Object released")
            return True

        except Exception as e:
            logger.error(f"Failed to release object: {e}")
            return False

    def lift_arm(self, amount: int = 50) -> bool:
        """
        Lift the robot arm

        Args:
            amount: Amount to lift (positive = up)

        Returns:
            bool: True if lift successful
        """
        try:
            if not self.arm:
                logger.warning("Arm not available")
                return True  # Non-critical

            logger.info(f"Lifting arm by {amount}...")
            self.arm.move(x=0, y=amount).wait_for_completed(timeout=3)
            time.sleep(0.3)
            return True

        except Exception as e:
            logger.warning(f"Arm lift failed: {e}")
            return True  # Non-critical

    def lower_arm(self, amount: int = 50) -> bool:
        """
        Lower the robot arm

        Args:
            amount: Amount to lower (positive value, will be negated)

        Returns:
            bool: True if lower successful
        """
        try:
            if not self.arm:
                logger.warning("Arm not available")
                return True  # Non-critical

            logger.info(f"Lowering arm by {amount}...")
            self.arm.move(x=0, y=-amount).wait_for_completed(timeout=3)
            time.sleep(0.3)
            return True

        except Exception as e:
            logger.warning(f"Arm lower failed: {e}")
            return True  # Non-critical

    def prepare_for_pickup(self) -> bool:
        """
        Prepare gripper for pickup (open wide).

        Returns:
            bool: True if preparation successful
        """
        try:
            if not self.gripper:
                logger.error("Gripper not available")
                return False

            logger.info("Opening gripper fully for pickup...")
            self.gripper.open(power=100)
            time.sleep(1.5)
            self.gripper.pause()

            logger.info("Gripper ready for pickup")
            return True

        except Exception as e:
            logger.error(f"Failed to prepare gripper: {e}")
            return False

    def sort_object(self, detected_object: DetectedObject,
                    approach_distance: float = 0.0,
                    movement_stack = None) -> bool:
        """
        Complete sorting pipeline for a single object.
        Assumes robot is already positioned at the object.

        Pipeline:
        1. Determine target zone using strategy
        2. Pick up object (gripper should already be open and positioned)
        3. Lift object
        4. Return to center (using movement_stack if provided)
        5. Navigate to target zone
        6. Release object
        7. Lower arm
        8. Return from zone to center

        Args:
            detected_object: Object to sort
            approach_distance: Distance traveled to reach object (legacy, for backward compat)
            movement_stack: MovementStack with recorded approach movements (preferred)

        Returns:
            bool: True if sorting successful
        """
        if not self.strategy:
            logger.error("No sorting strategy set")
            return False

        try:
            # Step 1: Determine target zone
            zone_name = self.strategy.determine_zone(detected_object)
            zone = self.zone_manager.get_zone(zone_name)

            if not zone:
                logger.error(f"Zone '{zone_name}' not found")
                return False

            if zone.is_full():
                logger.warning(f"Zone '{zone_name}' is full!")
                # Try default zone
                zone = self.zone_manager.get_zone("zone_default")
                if not zone or zone.is_full():
                    logger.error("All zones full")
                    return False

            logger.info(f"Sorting {detected_object.class_name} to {zone.name} ({zone.description})")

            # Step 2: Pick up object
            if not self.pick_up_object():
                logger.error("Failed to pick up object")
                return False

            # Step 3: Lift object
            self.lift_arm(50)

            # Step 4: Return from approach position first
            if movement_stack is not None and movement_stack.get_count() > 0:
                logger.info(f"Reversing {movement_stack.get_count()} movements to return to center...")
                movement_stack.reverse_all(self.chassis, xy_speed=self.approach_speed)
                movement_stack.clear()
                time.sleep(0.3)
            elif approach_distance > 0.1:
                # Fallback to simple backward movement (legacy)
                logger.info(f"Returning {approach_distance:.2f}m to center (simple backward)...")
                try:
                    self.chassis.move(x=-approach_distance, y=0, z=0,
                                    xy_speed=self.approach_speed).wait_for_completed(timeout=10)
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning(f"Return movement warning: {e}")

            # Step 5: Navigate to zone
            if not self.navigate_to_zone(zone):
                logger.error("Failed to navigate to zone")
                self.release_object()  # Drop object to recover
                return False

            # Step 6: Lower arm first (before releasing!)
            self.lower_arm(50)

            # Step 7: Release object (after arm is lowered)
            if not self.release_object():
                logger.error("Failed to release object")
                return False

            # Step 8: Update zone count
            zone.add_object(detected_object.class_name)

            # Step 9: Re-open gripper for next pickup
            self.prepare_for_pickup()

            # Step 10: Return to center
            if not self.return_from_zone(zone):
                logger.warning("Failed to return from zone cleanly")

            self.sorted_objects_count += 1
            logger.info(f"Successfully sorted {detected_object.class_name} to {zone.name} "
                       f"(Total: {self.sorted_objects_count})")
            return True

        except Exception as e:
            logger.error(f"Error sorting object: {e}")
            self.failed_count += 1
            # Try to recover
            try:
                self.release_object()
            except:
                pass
            return False

    def sort_objects_batch(self, detected_objects: List[DetectedObject]) -> int:
        """
        Sort multiple detected objects

        Args:
            detected_objects: List of objects to sort

        Returns:
            int: Number of successfully sorted objects
        """
        logger.info(f"Sorting batch of {len(detected_objects)} objects")
        sorted_count = 0

        for obj in detected_objects:
            if self.sort_object(obj):
                sorted_count += 1

        return sorted_count

    def get_sorting_statistics(self) -> dict:
        """Get statistics about sorting operations"""
        return {
            "total_sorted": self.sorted_objects_count,
            "total_failed": self.failed_count,
            "success_rate": (self.sorted_objects_count /
                           max(1, self.sorted_objects_count + self.failed_count) * 100),
            "strategy": self.strategy.__class__.__name__ if self.strategy else None,
            "zones": self.zone_manager.get_summary()
        }

    def reset_statistics(self):
        """Reset sorting statistics"""
        self.sorted_objects_count = 0
        self.failed_count = 0
        self.zone_manager.clear_all_zones()
        logger.info("Reset sorting statistics and cleared zones")

    def navigate_to_object(self, detected_object: DetectedObject,
                          frame_width: int, frame_height: int) -> Tuple[bool, float]:
        """
        Navigate robot towards detected object using visual servoing.

        Args:
            detected_object: Object to approach
            frame_width: Camera frame width
            frame_height: Camera frame height

        Returns:
            Tuple: (success, distance_traveled)
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available")
                return False, 0.0

            # Calculate angle offset from center of frame
            center_x, center_y = detected_object.center
            frame_center_x = frame_width / 2

            offset_x = center_x - frame_center_x
            # RoboMaster: negative angle = RIGHT, positive = LEFT
            # If object is RIGHT of center (+offset_x), turn RIGHT (-angle)
            angle_offset = -(offset_x / frame_center_x) * 30  # Scale to ~30° max

            logger.info(f"Object at ({center_x:.0f}, {center_y:.0f}), "
                       f"angle offset: {angle_offset:.1f}°")

            # Center on object first (if needed)
            if abs(angle_offset) > 2:
                logger.info(f"Centering on object ({angle_offset:.1f}°)...")
                self.chassis.move(x=0, y=0, z=angle_offset,
                                z_speed=30).wait_for_completed(timeout=3)
                time.sleep(0.3)

            return True, 0.0  # Will be extended with iterative approach

        except Exception as e:
            logger.error(f"Failed to navigate to object: {e}")
            return False, 0.0
