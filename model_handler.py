"""
Handle model loading and predictions.
Updated with Google Drive support.
"""

import numpy as np
import streamlit as st
import os
from typing import List, Tuple
from gdrive_utils import ensure_dataset_exists, ensure_model_exists


class ModelHandler:
    """Handle model loading and predictions."""

    def __init__(self, model_path: str, dataset_dir: str,
                 dataset_gdrive_id: str = None, model_gdrive_id: str = None):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.dataset_gdrive_id = dataset_gdrive_id
        self.model_gdrive_id = model_gdrive_id
        self.model = None
        self.class_labels = []

    def load_model(self):
        """Load the trained Keras model."""
        # Ensure model file exists (download if needed)
        if self.model_gdrive_id:
            model_available = ensure_model_exists(self.model_path, self.model_gdrive_id)
            if not model_available:
                st.sidebar.error("❌ Cannot load model - file not found")
                return False

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
        # Ensure dataset exists (download if needed)
        if self.dataset_gdrive_id and self.dataset_gdrive_id != "YOUR_FOLDER_ID_HERE":
            dataset_available = ensure_dataset_exists(
                self.dataset_dir,
                self.dataset_gdrive_id
            )
            if not dataset_available:
                st.sidebar.warning("⚠️ Using default A-Z classes (dataset not available)")
                self.class_labels = [chr(i) for i in range(65, 91)]
                return self.class_labels

        try:
            if os.path.exists(self.dataset_dir) and os.path.isdir(self.dataset_dir):
                # Get folder names as class labels
                classes = [
                    d for d in os.listdir(self.dataset_dir)
                    if os.path.isdir(os.path.join(self.dataset_dir, d))
                ]

                # Remove hidden folders (starting with .)
                classes = [c for c in classes if not c.startswith('.')]
                classes.sort()

                if classes:
                    self.class_labels = classes
                    st.sidebar.success(f"✅ Loaded {len(classes)} classes from dataset")
                    return self.class_labels
                else:
                    st.sidebar.warning("⚠️ Dataset folder empty, using default A-Z classes")
                    self.class_labels = [chr(i) for i in range(65, 91)]
                    return self.class_labels
            else:
                st.sidebar.warning(f"⚠️ Dataset not found: {self.dataset_dir}")
                st.sidebar.info("Using default A-Z classes")
                self.class_labels = [chr(i) for i in range(65, 91)]
                return self.class_labels

        except Exception as e:
            st.sidebar.error(f"❌ Error loading classes: {str(e)}")
            st.sidebar.info("Falling back to default A-Z classes")
            self.class_labels = [chr(i) for i in range(65, 91)]
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

            # Safety check
            if predictions.shape[1] != len(self.class_labels):
                st.warning(f"Model output size ({predictions.shape[1]}) != classes ({len(self.class_labels)})")

            predicted_class_idx = np.argmax(predictions[0])

            # Ensure index is valid
            if predicted_class_idx >= len(self.class_labels):
                predicted_class_idx = 0

            confidence = predictions[0][predicted_class_idx]
            predicted_label = self.class_labels[predicted_class_idx]

            return predicted_label, float(confidence), predictions[0]

        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "Error", 0.0, np.zeros(len(self.class_labels))