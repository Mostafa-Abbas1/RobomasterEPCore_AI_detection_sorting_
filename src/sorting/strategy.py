"""
Sorting Strategy Module
Defines different strategies for sorting objects
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..vision.detection import DetectedObject
import logging

logger = logging.getLogger(__name__)


class SortingStrategy(ABC):
    """
    Abstract base class for sorting strategies
    """

    @abstractmethod
    def determine_zone(self, detected_object: DetectedObject) -> str:
        """
        Determine which zone an object should be sorted into

        Args:
            detected_object: Object to sort

        Returns:
            str: Name of the target zone
        """
        pass

    @abstractmethod
    def get_priority(self, detected_object: DetectedObject) -> int:
        """
        Get sorting priority for an object (higher = more urgent)

        Args:
            detected_object: Object to evaluate

        Returns:
            int: Priority value
        """
        pass


class ClassBasedStrategy(SortingStrategy):
    """
    Sort objects based on their detected class
    """

    def __init__(self, class_zone_mapping: Optional[dict] = None):
        """
        Initialize class-based sorting strategy

        Args:
            class_zone_mapping: Dictionary mapping class names to zone names
        """
        self.class_zone_mapping = class_zone_mapping or {
            "cube": "zone_a",
            "sphere": "zone_b",
            "cylinder": "zone_c"
        }
        logger.info(f"Initialized ClassBasedStrategy with mapping: {self.class_zone_mapping}")

    def determine_zone(self, detected_object: DetectedObject) -> str:
        """
        Determine zone based on object class

        Args:
            detected_object: Object to sort

        Returns:
            str: Target zone name
        """
        zone = self.class_zone_mapping.get(detected_object.class_name, "zone_default")
        logger.info(f"Object class '{detected_object.class_name}' -> zone '{zone}'")
        return zone

    def get_priority(self, detected_object: DetectedObject) -> int:
        """
        All objects have equal priority in class-based sorting

        Args:
            detected_object: Object to evaluate

        Returns:
            int: Priority value (always 1)
        """
        return 1

    def add_class_mapping(self, class_name: str, zone_name: str):
        """
        Add or update a class to zone mapping

        Args:
            class_name: Name of the object class
            zone_name: Name of the target zone
        """
        self.class_zone_mapping[class_name] = zone_name
        logger.info(f"Added mapping: {class_name} -> {zone_name}")


class SizeBasedStrategy(SortingStrategy):
    """
    Sort objects based on their size
    """

    def __init__(self):
        """Initialize size-based sorting strategy"""
        self.size_thresholds = {
            "small": 1000,   # pixels²
            "medium": 5000,  # pixels²
            "large": float('inf')
        }
        logger.info("Initialized SizeBasedStrategy")

    def determine_zone(self, detected_object: DetectedObject) -> str:
        """
        Determine zone based on object size

        Args:
            detected_object: Object to sort

        Returns:
            str: Target zone name
        """
        # Calculate bounding box area
        x1, y1, x2, y2 = detected_object.bbox
        area = (x2 - x1) * (y2 - y1)

        if area < self.size_thresholds["small"]:
            zone = "zone_small"
        elif area < self.size_thresholds["medium"]:
            zone = "zone_medium"
        else:
            zone = "zone_large"

        logger.info(f"Object size {area}px² -> zone '{zone}'")
        return zone

    def get_priority(self, detected_object: DetectedObject) -> int:
        """
        Smaller objects get higher priority

        Args:
            detected_object: Object to evaluate

        Returns:
            int: Priority value (1-3, higher for smaller objects)
        """
        x1, y1, x2, y2 = detected_object.bbox
        area = (x2 - x1) * (y2 - y1)

        if area < self.size_thresholds["small"]:
            return 3
        elif area < self.size_thresholds["medium"]:
            return 2
        return 1


class ConfidenceBasedStrategy(SortingStrategy):
    """
    Sort objects based on detection confidence
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize confidence-based sorting strategy

        Args:
            confidence_threshold: Threshold for high confidence detections
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initialized ConfidenceBasedStrategy with threshold {confidence_threshold}")

    def determine_zone(self, detected_object: DetectedObject) -> str:
        """
        Determine zone based on detection confidence

        Args:
            detected_object: Object to sort

        Returns:
            str: Target zone name
        """
        if detected_object.confidence >= self.confidence_threshold:
            zone = "zone_high_confidence"
        else:
            zone = "zone_low_confidence"

        logger.info(f"Object confidence {detected_object.confidence:.2f} -> zone '{zone}'")
        return zone

    def get_priority(self, detected_object: DetectedObject) -> int:
        """
        Higher confidence objects get higher priority

        Args:
            detected_object: Object to evaluate

        Returns:
            int: Priority value based on confidence
        """
        return int(detected_object.confidence * 10)
