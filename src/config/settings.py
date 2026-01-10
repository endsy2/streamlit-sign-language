"""
Configuration settings for Sign Language Recognition System
"""


class Config:
    """Configuration settings for the application."""
    
    # Model settings
    MODEL_PATH = "models/sign_language_model.keras"
    
    # Detection settings
    SEQUENCE_LENGTH = 120  # 30 frames per record
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to show more results
    
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
    # MANUAL CATEGORIES - Define your sign language categories here
    # ============================================================================
    CLASS_LABELS = [
        'again', 'Baby', 'Bad', 'bathroom', 'book', 'Brother', 'busy', 'Dad', 
        'do not want', 'Eat', 'father', 'Fine', 'finish', 'forget', 'Go', 'Good', 
        'Great', 'happy', 'He', 'hello', 'Help', 'how', 'I', 'is', 'learn', 'like', 
        'Love', 'marry', 'meet', 'milk', 'more', 'mother', 'My', 'name', 'need', 
        'nice', 'No', 'Nothing', 'please', 'question', 'right', 'sad', 'same', 
        'Say', 'see you letter', 'Sister', 'sleep', 'Stop', 'thank you', 'want', 
        'We', 'what', 'What_s up', 'when', 'where', 'which', 'who', 'why', 
        'wrong', 'Yes', 'You', 'your'
    ]
