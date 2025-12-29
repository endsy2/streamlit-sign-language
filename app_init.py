"""
Sign Language Recognition System
Complete working version with proper import order
"""

# ============================================================================
# IMPORTS - MUST BE FIRST
# ============================================================================
import streamlit as st
from model_handler import ModelHandler


# ============================================================================
# APP INITIALIZATION FUNCTIONS
# ============================================================================
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


def setup_sidebar(config):
    """Setup sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")

    model_path = st.sidebar.text_input("Model Path", config.MODEL_PATH)
    dataset_dir = st.sidebar.text_input("Dataset Directory", config.DATASET_DIR)

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

    sequence_length = st.sidebar.number_input(
        "Sequence Length",
        min_value=1,
        max_value=60,
        value=config.SEQUENCE_LENGTH
    )

    show_keypoints = st.sidebar.checkbox("Show Keypoints", value=config.SHOW_KEYPOINTS)

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