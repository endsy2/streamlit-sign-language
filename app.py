"""
Sign Language Recognition System
Main Application Entry Point
"""
import streamlit as st
from src.config.settings import Config
from src.core.model_handler import ModelHandler
from src.tabs.webcam_tab import run_webcam_tab
from src.tabs.info_tab import run_info_tab


# Page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)


@st.cache_resource
def load_model(model_path: str, class_labels: list):
    """Load and cache model."""
    handler = ModelHandler(model_path)
    handler.load_model()
    handler.load_classes(class_labels)
    return handler


def setup_sidebar():
    """Setup sidebar configuration."""
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.subheader("🎯 Detection")
    min_detection = st.sidebar.slider("Detection Confidence", 0.0, 1.0, Config.MIN_DETECTION_CONFIDENCE, 0.05)
    min_tracking = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, Config.MIN_TRACKING_CONFIDENCE, 0.05)
    
    st.sidebar.subheader("📊 Model")
    st.sidebar.info(f"Frames: **{Config.SEQUENCE_LENGTH}**")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.CONFIDENCE_THRESHOLD, 0.05)
    show_keypoints = st.sidebar.checkbox("Show Keypoints", Config.SHOW_KEYPOINTS)
    
    return {
        'model_path': Config.MODEL_PATH,
        'sequence_length': Config.SEQUENCE_LENGTH,
        'min_detection_conf': min_detection,
        'min_tracking_conf': min_tracking,
        'confidence_threshold': confidence_threshold,
        'show_keypoints': show_keypoints
    }


def main():
    # Header
    st.title("🤟 Sign Language Recognition")
    st.markdown("Real-time sign language recognition using hand keypoints")
    
    # Sidebar
    settings = setup_sidebar()
    
    # Load model
    model_handler = load_model(Config.MODEL_PATH, Config.CLASS_LABELS)
    
    # Show categories
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Categories")
    st.sidebar.info(f"Total: {len(model_handler.class_labels)}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📷 Webcam", "ℹ️ Info"])
    
    with tab1:
        run_webcam_tab(settings, model_handler)
    
    with tab2:
        run_info_tab(settings, model_handler)


if __name__ == "__main__":
    main()
