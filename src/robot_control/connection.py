"""
RoboMaster Connection Module
Handles connecting and disconnecting from the RoboMaster EP Core
"""

from typing import Optional
import logging
from robomaster import robot

logger = logging.getLogger(__name__)


class RobotConnection:
    """
    Manages connection to the RoboMaster EP Core robot
    """

    def __init__(self):
        """Initialize the robot connection manager"""
        self.robot = None
        self.is_connected = False

    def connect(self, ip_address: Optional[str] = None, conn_type: str = "sta") -> bool:
        """
        Connect to the RoboMaster robot

        Args:
            ip_address: Optional IP address of the robot. If None, uses auto-discovery
            conn_type: Connection type - 'sta' for WiFi router, 'ap' for direct connection

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Attempting to connect to robot using {conn_type} mode...")

            # Create robot instance
            self.robot = robot.Robot()

            # Initialize connection (sta = WiFi router mode, ap = direct AP mode)
            self.robot.initialize(conn_type=conn_type)

            # Get version to verify connection
            version = self.robot.get_version()
            logger.info(f"Successfully connected! Robot version: {version}")

            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from the RoboMaster robot

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.robot and self.is_connected:
                logger.info("Disconnecting from robot...")
                self.robot.close()
                self.is_connected = False
                logger.info("Successfully disconnected")
                return True
            else:
                logger.warning("No active connection to disconnect")
                return False

        except Exception as e:
            logger.error(f"Error during disconnection: {e}")
            return False

    def is_robot_connected(self) -> bool:
        """
        Check if robot is currently connected

        Returns:
            bool: True if connected, False otherwise
        """
        return self.is_connected

    def get_robot_info(self) -> dict:
        """
        Get information about the connected robot

        Returns:
            dict: Robot information (version, battery, etc.)
        """
        if not self.is_connected or not self.robot:
            logger.warning("Cannot get robot info - not connected")
            return {}

        try:
            version = self.robot.get_version()
            sn = self.robot.get_sn()

            return {
                "version": version,
                "serial_number": sn,
                "connected": self.is_connected
            }
        except Exception as e:
            logger.error(f"Error getting robot info: {e}")
            return {"error": str(e)}
