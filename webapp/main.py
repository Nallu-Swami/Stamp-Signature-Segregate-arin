"""
Main entry point for the Streamlit YOLO application.
"""

import config
from modules.ui_manager import UIManager

def main():
    """Run the main application."""
    # Initialize directories
    config.initialize_directories()
    
    # Create and run the UI manager
    ui_manager = UIManager()
    ui_manager.run_app()

if __name__ == "__main__":
    main()
