import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple
import os

logger = logging.getLogger(__name__)

class StudentCounter:
    """
    Student counter using YOLOv8 for person detection
    
    This class uses a pre-trained YOLO model to detect people (students)
    in classroom photos. It does NOT do face recognition or identity extraction.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the student counter
        
        Args:
            model_name: YOLO model to use (yolov8n.pt is lightweight and fast)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        try:
            logger.info(f"Loading YOLO model: {model_name}")
            # YOLOv8 will auto-download the model if not present
            self.model = YOLO(model_name)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def count_students(self, image: Image.Image) -> Dict:
        """
        Count students in the image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with count, min_count, max_count, and detections
        """
        try:
            # Convert PIL image to numpy array (OpenCV format)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run YOLO inference
            results = self.model(img_array, conf=self.confidence_threshold, verbose=False)
            
            # Extract person detections (class 0 in COCO dataset is 'person')
            person_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        conf = float(box.conf[0])
                        person_detections.append({
                            "confidence": conf,
                            "bbox": box.xyxy[0].tolist()
                        })
            
            count = len(person_detections)
            
            # Calculate range with some tolerance
            # In a real scenario, this accounts for partial occlusions, edge cases
            uncertainty = max(1, int(count * 0.1))  # 10% uncertainty
            min_count = max(0, count - uncertainty)
            max_count = count + uncertainty
            
            avg_confidence = (
                sum(d["confidence"] for d in person_detections) / count
                if count > 0 else 0
            )
            
            logger.info(f"Detected {count} students with avg confidence {avg_confidence:.2f}")
            
            return {
                "count": count,
                "min_count": min_count,
                "max_count": max_count,
                "confidence": avg_confidence,
                "detections": person_detections
            }
            
        except Exception as e:
            logger.error(f"Error during student counting: {e}")
            # Return a safe default rather than failing
            return {
                "count": 0,
                "min_count": 0,
                "max_count": 0,
                "confidence": 0.0,
                "detections": [],
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model is not None,
            "confidence_threshold": self.confidence_threshold,
            "model_type": "YOLOv8"
        }
