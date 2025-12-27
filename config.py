"""
Sign Language Recognition System
Complete working version with proper import order
"""

# ============================================================================
# IMPORTS - MUST BE FIRST
# ============================================================================
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
from collections import deque, Counter
import tempfile
from typing import List, Tuple, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration settings for the application."""
    MODEL_PATH = "sign_language_model.h5"
    DATASET_DIR = "dataset"
    SEQUENCE_LENGTH = 30
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    MAX_NUM_HANDS = 2
    CONFIDENCE_THRESHOLD = 0.5
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    SHOW_KEYPOINTS = True
    SHOW_CONNECTIONS = True
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3
    KEYPOINTS_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
    TOTAL_KEYPOINTS = KEYPOINTS_PER_HAND * MAX_NUM_HANDS  # 126