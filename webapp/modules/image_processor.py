"""
Module for image processing with YOLO model.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
import supervision as sv
import config
from modules.storage_manager import StorageManager

class YOLOProcessor:
    """Class to handle YOLO model inference and image processing."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the YOLO processor.
        
        Args:
            model_path: Path to the YOLO model file
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self.storage_manager = StorageManager()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def load_model(self):
        """Load the YOLO model if not already loaded."""
        if self.model is None:
            self.model = YOLO(self.model_path)
        return self.model
    
    def process_image(self, image_path: str) -> Tuple[str, List[str]]:
        """
        Run YOLO inference on an image and return the annotated image and cropped objects.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
                - Path to the annotated output image
                - List of paths to extracted object images
        """
        # Load the model if not already loaded
        self.load_model()
        
        # Load the image and run inference
        image = cv2.imread(image_path)
        results = self.model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Annotate the image
        annotated_image = image.copy()
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Save the annotated image
        output_path = self.storage_manager.save_output_image(annotated_image)
        
        # Save and return cropped objects
        cropped_images = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(detections.xyxy):
            cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            cropped_path = self.storage_manager.save_extracted_object(cropped_image, i)
            cropped_images.append(cropped_path)
        
        return output_path, cropped_images
