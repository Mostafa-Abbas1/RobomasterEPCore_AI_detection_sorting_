"""
Object Detection Module
Handles AI model loading and object detection using YOLO or other models
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DetectedObject:
    """
    Represents a detected object in an image
    """

    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """
        Initialize detected object

        Args:
            class_name: Name of the detected object class
            confidence: Detection confidence score (0.0 to 1.0)
            bbox: Bounding box as (x1, y1, x2, y2)
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def __repr__(self):
        return f"DetectedObject(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class ObjectDetector:
    """
    Manages AI model loading and object detection using YOLOv8
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the object detector

        Args:
            model_path: Path to the trained model file (if None, uses pre-trained YOLOv8s)
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the AI detection model

        Args:
            model_path: Path to model file. If None, uses pre-trained YOLOv8s

        Returns:
            bool: True if model loaded successfully
        """
        try:
            from ultralytics import YOLO

            if model_path:
                self.model_path = model_path

            # If no model path specified, use pre-trained YOLOv8s
            if not self.model_path:
                logger.info("Loading pre-trained YOLOv8s model...")
                self.model = YOLO('yolov8s.pt')  # Small model (better than nano, still fast enough)
            else:
                logger.info(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)

            # Get class names from model
            self.class_names = list(self.model.names.values())
            logger.info(f"Model loaded successfully. Available classes: {len(self.class_names)}")
            logger.info(f"Classes: {self.class_names[:10]}...")  # Show first 10

            return True

        except ImportError:
            logger.error("Ultralytics YOLO not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in an image

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List[DetectedObject]: List of detected objects
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    # Create DetectedObject
                    detected_obj = DetectedObject(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    )
                    detections.append(detected_obj)

            logger.info(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []

    def detect_specific_class(self, image: np.ndarray, class_name: str) -> List[DetectedObject]:
        """
        Detect only objects of a specific class

        Args:
            image: Input image as numpy array
            class_name: Name of the class to detect

        Returns:
            List[DetectedObject]: List of detected objects of the specified class
        """
        all_detections = self.detect_objects(image)
        filtered = [obj for obj in all_detections if obj.class_name == class_name]
        logger.info(f"Found {len(filtered)} objects of class '{class_name}'")
        return filtered

    def draw_detections(self, image: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image

        Args:
            image: Input image as numpy array
            detections: List of detected objects to draw

        Returns:
            np.ndarray: Image with drawn detections
        """
        output_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with confidence
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw label background
            cv2.rectangle(
                output_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return output_image

    def get_supported_classes(self) -> List[str]:
        """
        Get list of classes the model can detect

        Returns:
            List[str]: List of class names
        """
        return self.class_names

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for detections

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Set confidence threshold to {self.confidence_threshold}")
