"""
RoboMaster Gripper Module
Handles EP gripper operations for object manipulation
"""

import time
import logging

logger = logging.getLogger(__name__)


class RobotGripper:
    """
    Controls the RoboMaster EP gripper for picking and placing objects
    """

    def __init__(self, robot):
        """
        Initialize the gripper controller

        Args:
            robot: Connected robot instance
        """
        self.robot = robot
        self.gripper = robot.gripper if robot else None
        self.is_open = True

    def open(self, power: int = 50) -> bool:
        """
        Open the gripper

        Args:
            power: Opening power (1-100)

        Returns:
            bool: True if gripper opened successfully
        """
        try:
            if not self.gripper:
                logger.error("Gripper not available - robot not connected")
                return False

            logger.info(f"Opening gripper with power {power}")
            self.gripper.open(power=power)
            time.sleep(1)  # Wait for gripper to open
            self.gripper.pause()
            self.is_open = True
            logger.info("Gripper opened successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to open gripper: {e}")
            return False

    def close(self, power: int = 50) -> bool:
        """
        Close the gripper

        Args:
            power: Closing power (1-100)

        Returns:
            bool: True if gripper closed successfully
        """
        try:
            if not self.gripper:
                logger.error("Gripper not available - robot not connected")
                return False

            logger.info(f"Closing gripper with power {power}")
            self.gripper.close(power=power)
            time.sleep(1)  # Wait for gripper to close
            self.gripper.pause()
            self.is_open = False
            logger.info("Gripper closed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to close gripper: {e}")
            return False

    def grab_object(self, power: int = 50) -> bool:
        """
        Attempt to grab an object with the gripper

        Args:
            power: Gripping power (1-100)

        Returns:
            bool: True if object grabbed successfully
        """
        try:
            logger.info("Attempting to grab object")

            # Close gripper to grab object
            success = self.close(power=power)

            if success:
                logger.info("Object grabbed")
                return True
            else:
                logger.warning("Failed to grab object")
                return False

        except Exception as e:
            logger.error(f"Error grabbing object: {e}")
            return False

    def release_object(self, power: int = 50) -> bool:
        """
        Release the currently held object

        Args:
            power: Opening power (1-100)

        Returns:
            bool: True if object released successfully
        """
        try:
            logger.info("Releasing object")

            # Open gripper to release object
            success = self.open(power=power)

            if success:
                logger.info("Object released")
                return True
            else:
                logger.warning("Failed to release object")
                return False

        except Exception as e:
            logger.error(f"Error releasing object: {e}")
            return False

    def is_holding_object(self) -> bool:
        """
        Check if gripper is currently holding an object

        Returns:
            bool: True if holding an object (gripper is closed)
        """
        # Note: Without force sensors, we assume gripper is holding if closed
        return not self.is_open

    def get_gripper_state(self) -> dict:
        """
        Get current gripper state information

        Returns:
            dict: Gripper state (open/closed, holding object, etc.)
        """
        return {
            "is_open": self.is_open,
            "is_holding": self.is_holding_object()
        }
