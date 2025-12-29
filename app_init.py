"""
Initialize Streamlit app with Google Drive support.
"""

import streamlit as st
from config import Config
from model_handler import ModelHandler


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Sign Language Recognition",
        page_icon="🤟",
        layout="wide"
    )


def display_header():
    """Display app title and description."""
    st.title("🤟 Sign Language Recognition System")
    st.markdown("Real-time sign language recognition using sequential keypoint tracking")

    # Show Google Drive status
    if 'dataset_downloaded' in st.session_state:
        st.info("📁 Dataset loaded from Google Drive")


def setup_sidebar(config):
    """Setup sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")

    # Google Drive settings
    st.sidebar.subheader("📥 Google Drive")

    # Show current IDs
    if config.DATASET_GDRIVE_ID != "YOUR_FOLDER_ID_HERE":
        st.sidebar.success("✅ Dataset ID configured")
    else:
        st.sidebar.error("❌ Dataset ID not configured")
        st.sidebar.info("Update DATASET_GDRIVE_ID in config.py")

    # Force re-download button
    if st.sidebar.button("🔄 Re-download Dataset"):
        import os
        import shutil
        from gdrive_utils import ensure_dataset_exists

        # Delete existing dataset
        if os.path.exists(config.DATASET_DIR):
            shutil.rmtree(config.DATASET_DIR)

        # Re-download
        ensure_dataset_exists(
            config.DATASET_DIR,
            config.DATASET_GDRIVE_ID,
            force_download=True
        )
        st.rerun()

    st.sidebar.markdown("---")

    # Model settings
    model_path = st.sidebar.text_input("Model Path", config.MODEL_PATH)
    dataset_dir = st.sidebar.text_input("Dataset Directory", config.DATASET_DIR)

    # Detection settings
    min_detection_conf = st.sidebar.slider(
        "Detection Confidence",
        0.0, 1.0,
        config.MIN_DETECTION_CONFIDENCE,
        0.05
    )
    min_tracking_conf = st.sidebar.slider(
        "Tracking Confidence",
        0.0, 1.0,
        config.MIN_TRACKING_CONFIDENCE,
        0.05
    )

    # Sequence settings
    sequence_length = st.sidebar.number_input(
        "Sequence Length",
        min_value=1,
        max_value=60,
        value=config.SEQUENCE_LENGTH
    )

    # Display settings
    show_keypoints = st.sidebar.checkbox("Show Keypoints", value=config.SHOW_KEYPOINTS)

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0,
        config.CONFIDENCE_THRESHOLD,
        0.05
    )

    return {
        'model_path': model_path,
        'dataset_dir': dataset_dir,
        'min_detection_conf': min_detection_conf,
        'min_tracking_conf': min_tracking_conf,
        'sequence_length': sequence_length,
        'show_keypoints': show_keypoints,
        'confidence_threshold': confidence_threshold
    }


@st.cache_resource
def init_model_handler(model_path, dataset_dir, dataset_gdrive_id, model_gdrive_id):
    """Initialize and cache model handler with Google Drive support."""
    handler = ModelHandler(
        model_path,
        dataset_dir,
        dataset_gdrive_id=dataset_gdrive_id,
        model_gdrive_id=model_gdrive_id
    )
    handler.load_model()
    handler.load_classes()
    return handler


def display_model_info(model_handler):
    """Display model information in sidebar."""
    st.sidebar.info(f"Classes: {len(model_handler.class_labels)}")
    if len(model_handler.class_labels) <= 10:
        st.sidebar.write("Loaded:", ", ".join(model_handler.class_labels))