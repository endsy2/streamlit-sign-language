"""
Main application with Google Drive dataset support.
"""

import streamlit as st
from config import Config
from app_init import (
    setup_page,
    display_header,
    setup_sidebar,
    init_model_handler,
    display_model_info
)
from webcam_tab import run_webcam_tab
from video_tab import run_video_tab
from info_tab import run_info_tab


def main():
    """Main application function."""
    setup_page()
    display_header()

    config = Config()
    settings = setup_sidebar(config)

    # Initialize model handler with Google Drive IDs
    model_handler = init_model_handler(
        settings['model_path'],
        settings['dataset_dir'],
        config.DATASET_GDRIVE_ID,
        config.MODEL_GDRIVE_ID
    )

    display_model_info(model_handler)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🎥 Video", "ℹ️ Info"])

    with tab1:
        run_webcam_tab(settings, model_handler)

    with tab2:
        run_video_tab(settings, model_handler)

    with tab3:
        run_info_tab(settings, model_handler)

    st.markdown("---")
    st.info("💡 Dataset automatically downloaded from Google Drive on first run!")


if __name__ == "__main__":
    main()