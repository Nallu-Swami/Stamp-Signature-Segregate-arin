"""
Module for managing file storage operations.
"""

import os
from typing import Optional
import config

class StorageManager:
    """Class to handle file storage operations."""
    
    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        Saves an uploaded file to the storage directory.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(config.STORAGE_PATH, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path
    
    @staticmethod
    def save_output_image(image, filename: str = "output.png") -> str:
        """
        Saves a processed image to the output directory.
        
        Args:
            image: OpenCV image
            filename: Name for the output file
            
        Returns:
            str: Path to the saved output file
        """
        output_path = os.path.join(config.OUTPUT_PATH, filename)
        import cv2
        cv2.imwrite(output_path, image)
        return output_path
    
    @staticmethod
    def save_extracted_object(image, index: int) -> str:
        """
        Saves an extracted object image to the extracted directory.
        
        Args:
            image: OpenCV image of the extracted object
            index: Index number for the extracted object
            
        Returns:
            str: Path to the saved extracted object image
        """
        extracted_path = os.path.join(config.EXTRACTED_PATH, f"object_{index}.png")
        import cv2
        cv2.imwrite(extracted_path, image)
        return extracted_path
