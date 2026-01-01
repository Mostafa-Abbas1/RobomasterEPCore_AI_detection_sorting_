"""
Object Tracking Module
Handles tracking of detected objects across multiple frames
"""

import numpy as np
from typing import List, Optional, Dict
from .detection import DetectedObject
import logging

logger = logging.getLogger(__name__)


class TrackedObject:
    """
    Represents an object being tracked across frames
    """

    def __init__(self, object_id: int, detected_object: DetectedObject):
        """
        Initialize tracked object

        Args:
            object_id: Unique identifier for this tracked object
            detected_object: Initial detection of the object
        """
        self.object_id = object_id
        self.class_name = detected_object.class_name
        self.current_bbox = detected_object.bbox
        self.confidence = detected_object.confidence
        self.history = [detected_object.bbox]
        self.frames_since_seen = 0

    def update(self, detected_object: DetectedObject):
        """
        Update tracked object with new detection

        Args:
            detected_object: New detection of the object
        """
        self.current_bbox = detected_object.bbox
        self.confidence = detected_object.confidence
        self.history.append(detected_object.bbox)
        self.frames_since_seen = 0

    def __repr__(self):
        return f"TrackedObject(id={self.object_id}, class={self.class_name}, bbox={self.current_bbox})"


class ObjectTracker:
    """
    Tracks objects across multiple frames
    """

    def __init__(self, max_disappeared: int = 10):
        """
        Initialize the object tracker

        Args:
            max_disappeared: Maximum frames an object can disappear before being removed
        """
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}

    def update(self, detections: List[DetectedObject]) -> List[TrackedObject]:
        """
        Update tracker with new detections

        Args:
            detections: List of detected objects in current frame

        Returns:
            List[TrackedObject]: List of currently tracked objects
        """
        # TODO: Implement tracking update logic
        logger.info(f"Updating tracker with {len(detections)} detections")
        return list(self.tracked_objects.values())

    def track_object(self, class_name: str) -> Optional[TrackedObject]:
        """
        Get the tracked object of a specific class

        Args:
            class_name: Name of the class to track

        Returns:
            Optional[TrackedObject]: Tracked object if found, None otherwise
        """
        # TODO: Implement class-specific tracking
        for tracked_obj in self.tracked_objects.values():
            if tracked_obj.class_name == class_name:
                return tracked_obj
        return None

    def get_object_by_id(self, object_id: int) -> Optional[TrackedObject]:
        """
        Get tracked object by its ID

        Args:
            object_id: ID of the object to retrieve

        Returns:
            Optional[TrackedObject]: Tracked object if found, None otherwise
        """
        return self.tracked_objects.get(object_id)

    def get_all_tracked_objects(self) -> List[TrackedObject]:
        """
        Get all currently tracked objects

        Returns:
            List[TrackedObject]: List of all tracked objects
        """
        return list(self.tracked_objects.values())

    def remove_object(self, object_id: int):
        """
        Remove an object from tracking

        Args:
            object_id: ID of the object to remove
        """
        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]
            logger.info(f"Removed tracked object {object_id}")

    def reset(self):
        """
        Reset the tracker, removing all tracked objects
        """
        self.tracked_objects.clear()
        self.next_object_id = 0
        logger.info("Tracker reset")

    def get_object_trajectory(self, object_id: int) -> List:
        """
        Get the movement history of a tracked object

        Args:
            object_id: ID of the object

        Returns:
            List: History of bounding boxes
        """
        tracked_obj = self.tracked_objects.get(object_id)
        if tracked_obj:
            return tracked_obj.history
        return []
