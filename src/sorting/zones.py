"""
Sorting Zones Module
Defines and manages sorting zones where objects are placed
"""

from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SortingZone:
    """
    Represents a physical zone where objects are sorted
    """

    def __init__(self, name: str, position: Tuple[float, float], capacity: int = 10,
                 navigation_angle: float = 0, return_angle: float = 0, description: str = ""):
        """
        Initialize a sorting zone

        Args:
            name: Name identifier for the zone
            position: Physical position as (x, y) coordinates in meters
            capacity: Maximum number of objects the zone can hold
            navigation_angle: Angle to rotate from center to face zone (degrees)
                             Positive = counter-clockwise (LEFT)
                             Negative = clockwise (RIGHT)
            return_angle: Angle to rotate to return to center orientation
            description: Human-readable description of the zone
        """
        self.name = name
        self.position = position
        self.capacity = capacity
        self.navigation_angle = navigation_angle
        self.return_angle = return_angle
        self.description = description
        self.current_count = 0
        self.objects_stored = []

    def is_full(self) -> bool:
        """
        Check if zone is at capacity

        Returns:
            bool: True if zone is full
        """
        return self.current_count >= self.capacity

    def add_object(self, object_class: str) -> bool:
        """
        Add an object to the zone

        Args:
            object_class: Class name of the object being added

        Returns:
            bool: True if object added successfully
        """
        if self.is_full():
            logger.warning(f"Zone '{self.name}' is full, cannot add object")
            return False

        self.objects_stored.append(object_class)
        self.current_count += 1
        logger.info(f"Added {object_class} to zone '{self.name}' ({self.current_count}/{self.capacity})")
        return True

    def remove_object(self, object_class: str) -> bool:
        """
        Remove an object from the zone

        Args:
            object_class: Class name of the object to remove

        Returns:
            bool: True if object removed successfully
        """
        if object_class in self.objects_stored:
            self.objects_stored.remove(object_class)
            self.current_count -= 1
            logger.info(f"Removed {object_class} from zone '{self.name}'")
            return True
        return False

    def clear(self):
        """
        Clear all objects from the zone
        """
        self.objects_stored.clear()
        self.current_count = 0
        logger.info(f"Cleared zone '{self.name}'")

    def get_info(self) -> dict:
        """
        Get information about the zone

        Returns:
            dict: Zone information
        """
        return {
            "name": self.name,
            "position": self.position,
            "capacity": self.capacity,
            "current_count": self.current_count,
            "available_space": self.capacity - self.current_count,
            "is_full": self.is_full(),
            "objects": self.objects_stored,
            "navigation_angle": self.navigation_angle,
            "return_angle": self.return_angle,
            "description": self.description
        }

    def __repr__(self):
        direction = "LEFT" if self.navigation_angle > 0 else "RIGHT" if self.navigation_angle < 0 else "STRAIGHT"
        return f"SortingZone(name={self.name}, {direction}, {self.current_count}/{self.capacity})"


class ZoneManager:
    """
    Manages multiple sorting zones
    """

    def __init__(self):
        """Initialize the zone manager"""
        self.zones: Dict[str, SortingZone] = {}
        logger.info("Initialized ZoneManager")

    def add_zone(self, zone: SortingZone):
        """
        Add a sorting zone

        Args:
            zone: SortingZone instance to add
        """
        self.zones[zone.name] = zone
        logger.info(f"Added zone: {zone}")

    def create_zone(self, name: str, position: Tuple[float, float], capacity: int = 10,
                    navigation_angle: float = 0, return_angle: float = 0,
                    description: str = "") -> SortingZone:
        """
        Create and add a new sorting zone

        Args:
            name: Name for the zone
            position: Physical position as (x, y)
            capacity: Maximum capacity
            navigation_angle: Angle to rotate to face zone
            return_angle: Angle to rotate to return to center
            description: Human-readable description

        Returns:
            SortingZone: The created zone
        """
        zone = SortingZone(name, position, capacity, navigation_angle, return_angle, description)
        self.add_zone(zone)
        return zone

    def get_zone(self, name: str) -> Optional[SortingZone]:
        """
        Get a zone by name

        Args:
            name: Name of the zone

        Returns:
            Optional[SortingZone]: The zone if found, None otherwise
        """
        return self.zones.get(name)

    def remove_zone(self, name: str) -> bool:
        """
        Remove a zone

        Args:
            name: Name of the zone to remove

        Returns:
            bool: True if zone removed successfully
        """
        if name in self.zones:
            del self.zones[name]
            logger.info(f"Removed zone '{name}'")
            return True
        return False

    def get_available_zones(self) -> List[SortingZone]:
        """
        Get all zones that are not full

        Returns:
            List[SortingZone]: List of available zones
        """
        return [zone for zone in self.zones.values() if not zone.is_full()]

    def get_all_zones(self) -> List[SortingZone]:
        """
        Get all zones

        Returns:
            List[SortingZone]: List of all zones
        """
        return list(self.zones.values())

    def clear_all_zones(self):
        """
        Clear all objects from all zones
        """
        for zone in self.zones.values():
            zone.clear()
        logger.info("Cleared all zones")

    def get_zone_by_position(self, position: Tuple[float, float], tolerance: float = 0.1) -> Optional[SortingZone]:
        """
        Find zone by approximate position

        Args:
            position: Target position as (x, y)
            tolerance: Position matching tolerance in meters

        Returns:
            Optional[SortingZone]: Closest zone within tolerance, None if not found
        """
        for zone in self.zones.values():
            distance = ((zone.position[0] - position[0]) ** 2 +
                       (zone.position[1] - position[1]) ** 2) ** 0.5
            if distance <= tolerance:
                return zone
        return None

    def get_summary(self) -> dict:
        """
        Get summary of all zones

        Returns:
            dict: Summary information
        """
        return {
            "total_zones": len(self.zones),
            "available_zones": len(self.get_available_zones()),
            "total_capacity": sum(z.capacity for z in self.zones.values()),
            "total_objects": sum(z.current_count for z in self.zones.values()),
            "zones": [zone.get_info() for zone in self.zones.values()]
        }
