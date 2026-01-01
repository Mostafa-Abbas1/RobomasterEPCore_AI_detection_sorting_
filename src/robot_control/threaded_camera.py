"""
Threaded Camera Module
Optimized camera reader for better real-time performance
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """
    Threaded camera reader that runs in background
    Always provides the latest frame without blocking
    """

    def __init__(self, ep_camera):
        """
        Initialize threaded camera

        Args:
            ep_camera: RoboMaster camera instance
        """
        self.ep_camera = ep_camera
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start background thread for frame reading"""
        logger.info("Starting threaded camera reader...")
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Give thread time to start
        logger.info("Threaded camera reader started")
        return self

    def _update(self):
        """
        Background thread - continuously reads frames
        This ensures we always have the latest frame available
        """
        logger.info("Background frame reader thread running")

        while not self.stopped:
            try:
                # Read frame from camera
                frame = self.ep_camera.read_cv2_image()

                if frame is not None:
                    # Update with latest frame
                    with self.lock:
                        self.frame = frame
                        self.frame_count += 1

            except Exception as e:
                logger.error(f"Error reading frame in background thread: {e}")

            # Small delay to prevent CPU overload
            time.sleep(0.01)

        logger.info("Background frame reader thread stopped")

    def read(self):
        """
        Get the latest available frame

        Returns:
            np.ndarray: Latest frame, or None if not available
        """
        with self.lock:
            return self.frame

    def stop(self):
        """Stop the background thread"""
        logger.info("Stopping threaded camera reader...")
        self.stopped = True

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        logger.info("Threaded camera reader stopped")

    def get_frame_count(self):
        """Get total number of frames read"""
        with self.lock:
            return self.frame_count

    def is_running(self):
        """Check if thread is running"""
        return self.thread is not None and self.thread.is_alive()
