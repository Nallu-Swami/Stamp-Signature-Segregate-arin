"""
Configuration settings for the Streamlit YOLO application.
"""

import os

# Define storage paths
STORAGE_PATH = 'storage/'
OUTPUT_PATH = 'output/'
EXTRACTED_PATH = 'extracted/'
MODEL_PATH = r"P:\WorkLAB\AIQoD\webapp2\model_weights\best.pt"

# Application settings
APP_TITLE = "📘 Minecraftors StampExtractor"
APP_DESCRIPTION = "Your Document Assistant"

# Create directories if they don't exist
def initialize_directories():
    """Create necessary directories if they don't exist."""
    for path in [STORAGE_PATH, OUTPUT_PATH, EXTRACTED_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
