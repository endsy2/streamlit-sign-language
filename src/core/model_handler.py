"""
Model Handler for Sign Language Recognition
"""
import streamlit as st
import numpy as np
from typing import List, Tuple


class ModelHandler:
    """Handle model loading and predictions."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.class_labels = []

    def load_model(self):
        """Load the trained Keras model."""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path)
            st.sidebar.success("✅ Model loaded!")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Model error: {str(e)}")
            return False

    def load_classes(self, categories: List[str]) -> List[str]:
        """Load class labels from manual config."""
        self.class_labels = categories if categories else ['Unknown']
        return self.class_labels

    def predict(self, sequence: List, sequence_length: int) -> Tuple[str, float, np.ndarray]:
        """Make prediction from keypoint sequence."""
        if len(sequence) < sequence_length:
            return "Collecting...", 0.0, np.zeros(len(self.class_labels))

        if self.model is None:
            return "Model not loaded", 0.0, np.zeros(len(self.class_labels))

        try:
            input_data = np.expand_dims(list(sequence), axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            label = self.class_labels[predicted_idx]
            return label, float(confidence), predictions[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "Error", 0.0, np.zeros(len(self.class_labels))
