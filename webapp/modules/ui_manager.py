"""
Module for managing the Streamlit UI components.
"""

import streamlit as st
from PIL import Image
from typing import Optional, Callable
import config
from modules.storage_manager import StorageManager
from modules.image_processor import YOLOProcessor

class UIManager:
    """Class to handle the Streamlit UI components and flow."""
    
    def __init__(self):
        """Initialize the UI Manager."""
        self.storage_manager = StorageManager()
        self.image_processor = YOLOProcessor()
        
    def setup_page(self):
        """Configure the page title and description."""
        st.title(config.APP_TITLE)
        st.markdown(f"### {config.APP_DESCRIPTION}")
        st.markdown("---")
    
    def create_upload_section(self) -> Optional[str]:
        """
        Create the file upload section.
        
        Returns:
            Optional[str]: Path to the saved file if uploaded, None otherwise
        """
        uploaded_png = st.file_uploader(
            "Upload Document (PNG)",
            type="png",
            help="Select a PNG document for analysis",
            accept_multiple_files=False
        )
        
        if uploaded_png:
            saved_path = self.storage_manager.save_uploaded_file(uploaded_png)
            st.success(f"✅ Document uploaded successfully! ({saved_path})")
            
            # Display the image preview
            image = Image.open(saved_path)
            st.image(image, caption="Uploaded Document Preview", use_container_width=True)
            
            return saved_path
        
        return None
    
    def create_inference_section(self, image_path: str):
        """
        Create the inference section with process button and results display.
        
        Args:
            image_path: Path to the uploaded image
        """
        if st.button("Proceed with Inference"):
            output_image_path, cropped_images = self.image_processor.process_image(image_path)
            st.success("✅ Inference completed successfully!")
            
            # Display the output image
            output_image = Image.open(output_image_path)
            st.image(output_image, caption="Processed Image", use_container_width=True)
            
            # Display extracted objects
            if cropped_images:
                st.markdown("### Extracted Annotations")
                for cropped_image_path in cropped_images:
                    extracted_image = Image.open(cropped_image_path)
                    st.image(extracted_image, caption="Extracted Object", use_container_width=True)
    
    def run_app(self):
        """Run the full application flow."""
        self.setup_page()
        uploaded_image_path = self.create_upload_section()
        
        if uploaded_image_path:
            self.create_inference_section(uploaded_image_path)
