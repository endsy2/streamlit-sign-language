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
            # Load without compiling to avoid optimizer version compatibility issues
            self.model = keras.models.load_model(self.model_path, compile=False)
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
            return "Collecting...", 0.0, np.zeros(len(self.class_labels) if self.class_labels else 1)

        if self.model is None:
            return "Model not loaded", 0.0, np.zeros(len(self.class_labels) if self.class_labels else 1)

        try:
            # Match the working test exactly
            X_input = np.array(list(sequence), dtype=np.float32)
            X_input = np.expand_dims(X_input, axis=0)  # shape (1, SEQUENCE_LENGTH, 126)
            
            pred = self.model.predict(X_input, verbose=0)
            pred_array = pred[0] if len(pred.shape) > 1 else pred
            pred_array = np.array(pred_array).flatten()
            
            predicted_idx = int(np.argmax(pred_array))
            confidence = float(np.max(pred_array))
            
            # Get label
            if self.class_labels and predicted_idx < len(self.class_labels):
                label = self.class_labels[predicted_idx]
            else:
                label = f"Class_{predicted_idx}"
            
            return label, confidence, pred_array
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return "Error", 0.0, np.zeros(len(self.class_labels) if self.class_labels else 1)
