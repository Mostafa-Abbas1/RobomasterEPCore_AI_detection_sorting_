"""
RoboMaster Camera Module
Handles camera streaming and image capture
"""

import cv2
import numpy as np
from typing import Optional
import logging
import threading
import time

logger = logging.getLogger(__name__)


class RobotCamera:
    """
    Manages the robot's camera for streaming and image capture
    """

    def __init__(self, robot):
        """
        Initialize the camera controller

        Args:
            robot: Connected robot instance
        """
        self.robot = robot
        self.camera = robot.camera if robot else None
        self.is_streaming = False
        self.current_frame = None

    def start_stream(self, display: bool = False) -> bool:
        """
        Start the camera video stream

        Args:
            display: Whether to display the stream in a window (not recommended for processing)

        Returns:
            bool: True if stream started successfully
        """
        try:
            if not self.camera:
                logger.error("Camera not available - robot not connected")
                return False

            logger.info("Starting camera stream...")
            self.camera.start_video_stream(display=display)
            self.is_streaming = True
            logger.info("Camera stream started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start camera stream: {e}")
            self.is_streaming = False
            return False

    def stop_stream(self) -> bool:
        """
        Stop the camera video stream

        Returns:
            bool: True if stream stopped successfully
        """
        try:
            if not self.camera:
                logger.error("Camera not available")
                return False

            if self.is_streaming:
                logger.info("Stopping camera stream...")
                self.camera.stop_video_stream()
                self.is_streaming = False
                logger.info("Camera stream stopped")
                return True
            else:
                logger.warning("Camera stream is not running")
                return False

        except Exception as e:
            logger.error(f"Failed to stop camera stream: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current camera frame

        Returns:
            Optional[np.ndarray]: Current frame as numpy array (BGR format), None if unavailable
        """
        try:
            if not self.is_streaming:
                logger.warning("Camera stream not running. Call start_stream() first.")
                return None

            if not self.camera:
                logger.error("Camera not available")
                return None

            # Read frame from camera
            frame = self.camera.read_cv2_image()
            self.current_frame = frame
            return frame

        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            return None

    def capture_image(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Capture a single image from the camera

        Args:
            save_path: Optional path to save the image

        Returns:
            Optional[np.ndarray]: Captured image as numpy array
        """
        try:
            frame = self.get_frame()

            if frame is None:
                logger.error("Failed to capture image - no frame available")
                return None

            if save_path:
                cv2.imwrite(save_path, frame)
                logger.info(f"Image saved to {save_path}")

            return frame

        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            bool: True if resolution set successfully
        """
        # Note: RoboMaster SDK uses predefined resolutions
        # Available: STREAM_360P, STREAM_540P, STREAM_720P
        logger.warning("RoboMaster SDK uses predefined resolutions. Custom resolution not supported.")
        logger.info(f"Requested resolution: {width}x{height}")
        return False

    def get_camera_info(self) -> dict:
        """
        Get camera information and settings

        Returns:
            dict: Camera information (resolution, fps, etc.)
        """
        return {
            "is_streaming": self.is_streaming,
            "has_current_frame": self.current_frame is not None,
            "frame_shape": self.current_frame.shape if self.current_frame is not None else None
        }
