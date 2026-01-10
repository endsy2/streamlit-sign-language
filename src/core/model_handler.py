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
        self.input_shape = None

    def load_model(self):
        """Load the trained Keras model."""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path)
            self.input_shape = self.model.input_shape
            st.sidebar.success("✅ Model loaded!")
            st.sidebar.info(f"Input: {self.input_shape}")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ Model error: {str(e)}")
            import traceback
            st.sidebar.code(traceback.format_exc())
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
            # Convert sequence to numpy array
            input_data = np.array(list(sequence), dtype=np.float32)
            
            # Get expected shape from model
            expected_shape = self.model.input_shape  # (None, seq_len, features)
            expected_seq_len = expected_shape[1] if expected_shape[1] else sequence_length
            expected_features = expected_shape[2] if len(expected_shape) > 2 else input_data.shape[-1]
            
            # Reshape if needed
            if len(input_data.shape) == 1:
                # Flat array - reshape to (seq_len, features)
                total_features = len(input_data) // sequence_length
                input_data = input_data.reshape(sequence_length, total_features)
            
            # Adjust sequence length if model expects different
            if input_data.shape[0] != expected_seq_len:
                if input_data.shape[0] > expected_seq_len:
                    input_data = input_data[-expected_seq_len:]  # Take last N frames
                else:
                    # Pad with zeros
                    padding = np.zeros((expected_seq_len - input_data.shape[0], input_data.shape[1]))
                    input_data = np.vstack([padding, input_data])
            
            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)
            
            # Make prediction
            predictions = self.model.predict(input_data, verbose=0)
            
            # Handle different output shapes
            if len(predictions.shape) > 1:
                pred_array = predictions[0]
            else:
                pred_array = predictions
            
            pred_array = np.array(pred_array).flatten()
            
            # Get prediction
            predicted_idx = int(np.argmax(pred_array))
            confidence = float(pred_array[predicted_idx])
            
            # Get label
            if self.class_labels and predicted_idx < len(self.class_labels):
                label = self.class_labels[predicted_idx]
            else:
                label = f"Class_{predicted_idx}"
            
            # Pad pred_array if needed to match class_labels length for display
            if len(self.class_labels) > len(pred_array):
                padded = np.zeros(len(self.class_labels))
                padded[:len(pred_array)] = pred_array
                pred_array = padded
            
            return label, confidence, pred_array
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return "Error", 0.0, np.zeros(len(self.class_labels) if self.class_labels else 1)
