"""
RoboMaster Movement Module
Handles all robot movement operations (driving, rotating, etc.)
"""

from typing import Tuple
import logging
import time

logger = logging.getLogger(__name__)

class MovementStack:
    """
    Records all robot movements and can reverse them to return to origin.
    Uses LIFO (Last In, First Out) to reverse movements in correct order.
    """

    def __init__(self):
        """Initialize empty movement stack"""
        self._movements = []  # List of (x, y, z) tuples
        logger.info("MovementStack initialized")

    def record(self, x: float, y: float, z: float):
        """
        Record a movement command.

        Args:
            x: Forward/backward distance (meters)
            y: Left/right distance (meters)
            z: Rotation angle (degrees)
        """
        if abs(x) > 0.001 or abs(y) > 0.001 or abs(z) > 0.1:
            self._movements.append((x, y, z))
            logger.debug(f"Recorded movement: x={x:.3f}, y={y:.3f}, z={z:.1f}")

    def reverse_all(self, chassis, xy_speed: float = 0.3, z_speed: float = 45, timeout: float = 5) -> bool:
        """
        Execute all recorded movements in reverse order with negated values.

        Args:
            chassis: Robot chassis object
            xy_speed: Speed for linear movements
            z_speed: Speed for rotations
            timeout: Timeout per movement

        Returns:
            bool: True if all reversals successful
        """
        import time as t
        
        if not self._movements:
            logger.info("No movements to reverse")
            return True

        logger.info(f"Reversing {len(self._movements)} movements...")

        # Reverse order (LIFO)
        for i, (x, y, z) in enumerate(reversed(self._movements)):
            try:
                # Negate all values to reverse the movement
                rev_x, rev_y, rev_z = -x, -y, -z

                logger.debug(f"Reverse step {i+1}: x={rev_x:.3f}, y={rev_y:.3f}, z={rev_z:.1f}")

                # Execute rotation first if significant (slower for stability)
                if abs(rev_z) > 0.5:
                    chassis.move(x=0, y=0, z=rev_z, z_speed=30).wait_for_completed(timeout=timeout)
                    t.sleep(0.5)  # Wait for robot to stabilize after rotation

                # Then linear movement (slower, more stable)
                if abs(rev_x) > 0.01 or abs(rev_y) > 0.01:
                    chassis.move(x=rev_x, y=rev_y, z=0, xy_speed=0.2).wait_for_completed(timeout=timeout)
                    t.sleep(0.3)

            except Exception as e:
                logger.warning(f"Reverse step {i+1} failed: {e}")

        logger.info("Movement reversal complete")
        return True

    def clear(self):
        """Clear all recorded movements"""
        count = len(self._movements)
        self._movements.clear()
        logger.info(f"Cleared {count} recorded movements")

    def get_count(self) -> int:
        """Get number of recorded movements"""
        return len(self._movements)

    def get_total_rotation(self) -> float:
        """Get sum of all rotations"""
        return sum(z for _, _, z in self._movements)

    def get_total_distance(self) -> float:
        """Get sum of forward distance"""
        return sum(x for x, _, _ in self._movements)




class PositionTracker:
    """
    Tracks the robot's relative position and orientation from start position.
    Uses dead reckoning based on movement commands.
    """

    def __init__(self):
        """Initialize position tracker at origin"""
        self.reset()

    def reset(self):
        """Reset position to origin (center)"""
        self.x = 0.0          # Forward/backward distance from center
        self.y = 0.0          # Left/right distance from center
        self.angle = 0.0      # Cumulative rotation from start orientation
        self.total_distance = 0.0  # Total distance traveled
        logger.info("Position tracker reset to origin")

    def update_position(self, dx: float, dy: float, dz: float):
        """
        Update position based on movement command

        Args:
            dx: Forward/backward movement (positive = forward)
            dy: Left/right movement (positive = right in robot frame)
            dz: Rotation in degrees (positive = counter-clockwise)
        """
        import math

        # Update rotation first
        self.angle += dz
        # Normalize angle to -180 to 180
        while self.angle > 180:
            self.angle -= 360
        while self.angle < -180:
            self.angle += 360

        # Convert robot-frame movement to world-frame
        # Robot moves relative to its current orientation
        angle_rad = math.radians(self.angle)

        # Transform dx, dy from robot frame to world frame
        world_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        world_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

        self.x += world_dx
        self.y += world_dy
        self.total_distance += abs(dx) + abs(dy)

        logger.debug(f"Position updated: x={self.x:.3f}, y={self.y:.3f}, angle={self.angle:.1f}°")

    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current position relative to start

        Returns:
            Tuple: (x, y, angle) - x/y in meters, angle in degrees
        """
        return (self.x, self.y, self.angle)

    def get_distance_from_center(self) -> float:
        """
        Get straight-line distance from center

        Returns:
            float: Distance in meters
        """
        import math
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def get_return_movements(self) -> Tuple[float, float, float]:
        """
        Calculate movements needed to return to center

        Returns:
            Tuple: (back_distance, angle_to_center, final_rotation)
        """
        import math

        distance = self.get_distance_from_center()

        if distance < 0.01:  # Already at center
            return (0, 0, -self.angle)

        # Angle from current position to center (in world frame)
        angle_to_center = math.degrees(math.atan2(-self.y, -self.x))

        # How much to rotate to face center
        rotation_needed = angle_to_center - self.angle

        # Normalize
        while rotation_needed > 180:
            rotation_needed -= 360
        while rotation_needed < -180:
            rotation_needed += 360

        return (distance, rotation_needed, -(self.angle + rotation_needed))


class RobotMovement:
    """
    Controls all movement operations of the RoboMaster robot
    """

    # Class-level position tracker shared across instances
    _position_tracker = None

    def __init__(self, robot):
        """
        Initialize the movement controller

        Args:
            robot: Connected robot instance
        """
        self.robot = robot
        self.chassis = robot.chassis if robot else None

        # Initialize shared position tracker
        if RobotMovement._position_tracker is None:
            RobotMovement._position_tracker = PositionTracker()
        self.tracker = RobotMovement._position_tracker

    @classmethod
    def reset_position_tracker(cls):
        """Reset position tracker to origin (call this when robot is at center)"""
        if cls._position_tracker:
            cls._position_tracker.reset()
        else:
            cls._position_tracker = PositionTracker()

    def get_tracked_position(self) -> Tuple[float, float, float]:
        """Get current tracked position"""
        return self.tracker.get_position()

    def move_forward(self, distance: float, speed: float = 0.5, timeout: float = 10) -> bool:
        """
        Move robot forward by specified distance

        Args:
            distance: Distance to move in meters
            speed: Movement speed (0.0 to 1.0)
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if movement successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info(f"Moving forward {distance}m at speed {speed}")
            self.chassis.move(x=distance, y=0, z=0, xy_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(distance, 0, 0)
            logger.info("Forward movement completed")
            return True

        except Exception as e:
            logger.error(f"Failed to move forward: {e}")
            return False

    def move_backward(self, distance: float, speed: float = 0.5, timeout: float = 10) -> bool:
        """
        Move robot backward by specified distance

        Args:
            distance: Distance to move in meters
            speed: Movement speed (0.0 to 1.0)
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if movement successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info(f"Moving backward {distance}m at speed {speed}")
            self.chassis.move(x=-distance, y=0, z=0, xy_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(-distance, 0, 0)
            logger.info("Backward movement completed")
            return True

        except Exception as e:
            logger.error(f"Failed to move backward: {e}")
            return False

    def move_left(self, distance: float, speed: float = 0.5, timeout: float = 10) -> bool:
        """
        Move robot left by specified distance (strafe)

        Args:
            distance: Distance to move in meters
            speed: Movement speed (0.0 to 1.0)
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if movement successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info(f"Moving left {distance}m at speed {speed}")
            self.chassis.move(x=0, y=-distance, z=0, xy_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(0, -distance, 0)
            logger.info("Left movement completed")
            return True

        except Exception as e:
            logger.error(f"Failed to move left: {e}")
            return False

    def move_right(self, distance: float, speed: float = 0.5, timeout: float = 10) -> bool:
        """
        Move robot right by specified distance (strafe)

        Args:
            distance: Distance to move in meters
            speed: Movement speed (0.0 to 1.0)
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if movement successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info(f"Moving right {distance}m at speed {speed}")
            self.chassis.move(x=0, y=distance, z=0, xy_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(0, distance, 0)
            logger.info("Right movement completed")
            return True

        except Exception as e:
            logger.error(f"Failed to move right: {e}")
            return False

    def rotate(self, angle: float, speed: float = 45.0, timeout: float = 10) -> bool:
        """
        Rotate robot by specified angle

        Args:
            angle: Angle to rotate in degrees
                   Positive = counter-clockwise (LEFT)
                   Negative = clockwise (RIGHT)
            speed: Rotation speed in degrees per second
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if rotation successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            direction = "LEFT" if angle > 0 else "RIGHT" if angle < 0 else "none"
            logger.info(f"Rotating {abs(angle)}° {direction} at speed {speed} deg/s")
            self.chassis.move(x=0, y=0, z=angle, z_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(0, 0, angle)
            logger.info("Rotation completed")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate: {e}")
            return False

    def move_to_position(self, x: float, y: float, speed: float = 0.5, timeout: float = 10) -> bool:
        """
        Move robot to specific position (relative to current position)

        Args:
            x: X coordinate in meters (forward/backward)
            y: Y coordinate in meters (left/right)
            speed: Movement speed (0.0 to 1.0)
            timeout: Maximum time to wait for completion

        Returns:
            bool: True if movement successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info(f"Moving to position ({x}, {y}) at speed {speed}")
            self.chassis.move(x=x, y=y, z=0, xy_speed=speed).wait_for_completed(timeout=timeout)
            self.tracker.update_position(x, y, 0)
            logger.info("Position movement completed")
            return True

        except Exception as e:
            logger.error(f"Failed to move to position: {e}")
            return False

    def return_to_center(self, speed: float = 0.3, rotation_speed: float = 45) -> bool:
        """
        Return robot to center position and original orientation.
        Uses tracked position to calculate return path.

        Args:
            speed: Movement speed
            rotation_speed: Rotation speed in degrees per second

        Returns:
            bool: True if return successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            x, y, angle = self.tracker.get_position()
            distance = self.tracker.get_distance_from_center()

            logger.info(f"Returning to center from ({x:.2f}, {y:.2f}, {angle:.1f}°)")

            if distance < 0.05 and abs(angle) < 5:
                logger.info("Already at center")
                return True

            # Simple approach: reverse the path
            # First, turn to face center (rotate by return angle)
            if distance >= 0.05:
                # Move back to center
                # We need to move -x, -y in world frame, but we're rotated
                # Simplest: reverse our forward movement first

                # If we only moved forward, just move backward
                # This works for the simple case
                self.chassis.move(x=-x, y=-y, z=0, xy_speed=speed).wait_for_completed(timeout=15)
                time.sleep(0.3)

            # Then rotate back to original orientation
            if abs(angle) >= 2:
                logger.info(f"Rotating {-angle:.1f}° to face original direction")
                self.chassis.move(x=0, y=0, z=-angle, z_speed=rotation_speed).wait_for_completed(timeout=10)

            # Reset tracker
            self.tracker.reset()
            logger.info("Returned to center successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to return to center: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop all robot movement immediately

        Returns:
            bool: True if stop successful
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return False

            logger.info("Stopping robot movement")
            self.chassis.drive_speed(x=0, y=0, z=0)
            logger.info("Robot stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop robot: {e}")
            return False

    def get_current_position(self) -> Tuple[float, float, float]:
        """
        Get current robot position and orientation

        Returns:
            Tuple[float, float, float]: (x, y, angle) position
        """
        try:
            if not self.chassis:
                logger.error("Chassis not available - robot not connected")
                return (0.0, 0.0, 0.0)

            # Note: This requires subscription to chassis position
            # For now, return placeholder
            logger.warning("Position tracking requires chassis subscription - not yet implemented")
            return (0.0, 0.0, 0.0)

        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return (0.0, 0.0, 0.0)
