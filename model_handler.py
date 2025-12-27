
"""
Sign Language Recognition System
Complete working version with proper import order
"""

# ============================================================================
# IMPORTS - MUST BE FIRST
# ============================================================================
import streamlit as st
import numpy as np
import os
from typing import List, Tuple
# ============================================================================
# MODEL HANDLER
# ============================================================================
class ModelHandler:
    """Handle model loading and predictions."""

    def __init__(self, model_path: str, dataset_dir: str):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.model = None
        self.class_labels = []

    def load_model(self):
        """Load the trained Keras model."""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path)
            st.sidebar.success("✅ Model loaded successfully!")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return False

    def load_classes(self) -> List[str]:
        """Load class names from dataset directory."""
        try:
            if os.path.exists(self.dataset_dir):
                classes = [
                    d for d in os.listdir(self.dataset_dir)
                    if os.path.isdir(os.path.join(self.dataset_dir, d))
                ]
                classes.sort()
                self.class_labels = classes
                return classes
            else:
                st.sidebar.warning(f"Dataset directory not found: {self.dataset_dir}")
                self.class_labels = [chr(i) for i in range(65, 91)]  # A-Z
                return self.class_labels
        except Exception as e:
            st.sidebar.error(f"Error loading classes: {e}")
            self.class_labels = ['A', 'B', 'C']
            return self.class_labels

    def predict(self, sequence: List, sequence_length: int) -> Tuple[str, float, np.ndarray]:
        """Make prediction from keypoint sequence."""
        if len(sequence) < sequence_length:
            return "Collecting frames...", 0.0, np.zeros(len(self.class_labels))

        if self.model is None:
            return "Model not loaded", 0.0, np.zeros(len(self.class_labels))

        try:
            input_data = np.expand_dims(list(sequence), axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_label = self.class_labels[predicted_class_idx]

            return predicted_label, float(confidence), predictions[0]

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "Error", 0.0, np.zeros(len(self.class_labels))