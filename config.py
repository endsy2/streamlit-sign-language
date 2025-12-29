"""
Configuration settings for the sign language recognition app.
Modify these settings to customize the application behavior.
"""

class Config:
    # Model settings
    MODEL_PATH = "sign_language_model.h5"
    DATASET_DIR = "dataset"

    # Sequence settings
    SEQUENCE_LENGTH = 30  # Number of frames for temporal sequence

    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    MAX_NUM_HANDS = 2

    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for valid prediction

    # Video settings
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # UI settings
    SHOW_KEYPOINTS = True
    SHOW_CONNECTIONS = True

    # Keypoint structure
    NUM_LANDMARKS = 21  # MediaPipe hand landmarks
    COORDS_PER_LANDMARK = 3  # x, y, z
    KEYPOINTS_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
    TOTAL_KEYPOINTS = KEYPOINTS_PER_HAND * MAX_NUM_HANDS  # 126