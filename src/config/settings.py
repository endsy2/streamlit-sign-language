"""
Configuration settings for Sign Language Recognition System
"""


class Config:
    """Configuration settings for the application."""
    
    # Model settings
    MODEL_PATH = "models/sign_language_model.keras"
    
    # Detection settings
    SEQUENCE_LENGTH = 30  # Must match training - 30 frames per prediction
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    MAX_NUM_HANDS = 2
    CONFIDENCE_THRESHOLD = 0.3
    
    # Camera settings
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    
    # Display settings
    SHOW_KEYPOINTS = True
    SHOW_CONNECTIONS = True
    
    # Keypoint settings
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3
    KEYPOINTS_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
    TOTAL_KEYPOINTS = KEYPOINTS_PER_HAND * MAX_NUM_HANDS  # 126
    
    # ============================================================================
    # CLASS LABELS - Must match training order exactly!
    # ============================================================================
    CLASS_LABELS = [
        "Baby", "Bad", "Brother", "Dad", "Eat", "Fine", "Friend", "Go", "Good", 
        "Great", "He", "Help", "I", "Love", "My", "No", "Nothing", "Say", 
        "See you later", "Sister", "Stop", "Teacher", "We", "What_s up", "Yes", 
        "You", "again", "bathroom", "book", "busy", "do not want", "father", 
        "finish", "forget", "happy", "hello", "how", "is", "learn", "like", 
        "marry", "meet", "milk", "more", "mother", "name", "need", "nice", 
        "please", "question", "right", "sad", "same", "see you letter", "sleep", 
        "thank you", "want", "what", "when", "where", "which", "who", "why", 
        "wrong", "your"
    ]
