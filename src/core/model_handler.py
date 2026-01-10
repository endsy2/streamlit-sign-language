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
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
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
            input_data = np.array(list(sequence))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Debug: show input shape
            # Expected shape: (1, sequence_length, 126)
            print(f"Input shape: {input_data.shape}")
            print(f"Model expected: {self.model.input_shape}")
            
            predictions = self.model.predict(input_data, verbose=0)
            
            # Handle different output shapes
            if len(predictions.shape) > 1:
                pred_array = predictions[0]
            else:
                pred_array = predictions
            
            # Check if predictions match class labels
            if len(pred_array) != len(self.class_labels):
                st.warning(f"⚠️ Model outputs {len(pred_array)} classes, but {len(self.class_labels)} labels defined")
                # Use model output size
                predicted_idx = np.argmax(pred_array)
                confidence = float(pred_array[predicted_idx])
                if predicted_idx < len(self.class_labels):
                    label = self.class_labels[predicted_idx]
                else:
                    label = f"Class_{predicted_idx}"
            else:
                predicted_idx = np.argmax(pred_array)
                confidence = float(pred_array[predicted_idx])
                label = self.class_labels[predicted_idx]
            
            return label, confidence, pred_array
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return "Error", 0.0, np.zeros(len(self.class_labels))
