
"""
Sign Language Recognition System
Complete working version with proper import order
"""

# ============================================================================
# IMPORTS - MUST BE FIRST
# ============================================================================
import streamlit as st

from app_init import setup_page, display_header, setup_sidebar
from config import Config
from info_tab import run_info_tab
from model_handler import ModelHandler
from video_tab import run_video_tab
from webcam_tab import run_webcam_tab


# ============================================================================
# MAIN APPLICATION
# ============================================================================



@st.cache_resource
def init_model_handler(model_path, dataset_dir):
    """Initialize and cache model handler."""
    handler = ModelHandler(model_path, dataset_dir)
    handler.load_model()
    handler.load_classes()
    return handler

def display_model_info(model_handler):
    """Display model information in sidebar."""
    st.sidebar.info(f"Classes: {len(model_handler.class_labels)}")
    if len(model_handler.class_labels) <= 10:
        st.sidebar.write("Loaded:", ", ".join(model_handler.class_labels))

def main():
    """Main application function."""
    # Setup page
    setup_page()

    # Display header
    display_header()

    # Initialize configuration
    config = Config()

    # Setup sidebar and get user settings
    settings = setup_sidebar(config)

    # Initialize model handler
    model_handler = init_model_handler(
        settings['model_path'],
        settings['dataset_dir']
    )

    # Display class info in sidebar
    display_model_info(model_handler)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🎥 Video", "ℹ️ Info"])

    # Run tabs
    with tab1:
        run_webcam_tab(settings, model_handler)

    with tab2:
        run_video_tab(settings, model_handler)

    with tab3:
        run_info_tab(settings, model_handler)

    # Footer
    st.markdown("---")
    st.info("💡 **Modular Design:** Easy to debug and maintain!")


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()