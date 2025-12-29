
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
from typing import List

# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    """Reusable UI components for consistent interface."""

    @staticmethod
    def display_prediction(label: str, confidence: float, threshold: float = 0.5):
        """Display prediction with appropriate styling."""
        if confidence > threshold:
            st.success(f"**Prediction:** {label}")
        elif confidence > threshold * 0.6:
            st.warning(f"**Prediction:** {label}")
        else:
            st.info(f"**Prediction:** {label} (Low confidence)")

    @staticmethod
    def display_confidence_metrics(confidence: float, hands_detected: List[str]):
        """Display confidence and hand detection metrics."""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{confidence:.2%}")
        with col2:
            hands_text = ", ".join(hands_detected) if hands_detected else "None"
            st.metric("Hands", hands_text)

    @staticmethod
    def display_top_predictions(predictions: np.ndarray,
                                class_labels: List[str],
                                top_k: int = 5):
        """Display bar chart of top predictions."""
        if predictions.sum() == 0:
            return

        # Safety check: ensure predictions match class_labels length
        if len(predictions) != len(class_labels):
            st.warning(
                f"⚠️ Mismatch: Model outputs {len(predictions)} classes but only {len(class_labels)} labels loaded")
            # Trim predictions to match class_labels
            predictions = predictions[:len(class_labels)]

        # Get top k predictions
        top_k = min(top_k, len(class_labels))  # Don't ask for more than available
        top_k_idx = np.argsort(predictions)[-top_k:][::-1]

        # Safety check: ensure indices are valid
        top_k_idx = [i for i in top_k_idx if i < len(class_labels)]

        if not top_k_idx:
            return

        top_k_labels = [class_labels[i] for i in top_k_idx]
        top_k_values = [float(predictions[i]) for i in top_k_idx]

        st.caption("Top Predictions")
        chart_data = {label: val for label, val in zip(top_k_labels, top_k_values)}
        st.bar_chart(chart_data)

    @staticmethod
    def display_sequence_progress(current: int, total: int, text: str = "Sequence"):
        """Display sequence collection progress bar."""
        progress = min(current / total, 1.0)
        st.progress(progress, text=f"{text}: {current}/{total} frames")

    @staticmethod
    def display_frame_with_overlay(frame: np.ndarray,
                                   label: str,
                                   confidence: float,
                                   sequence_count: int,
                                   sequence_length: int,
                                   hands_detected: List[str],
                                   threshold: float = 0.5):
        """Add text overlays to frame."""
        frame_copy = frame.copy()

        # Add prediction text
        if sequence_count >= sequence_length:
            color = (0, 255, 0) if confidence > threshold else (255, 165, 0)
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.putText(frame_copy, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            text = f"Collecting: {sequence_count}/{sequence_length}"
            cv2.putText(frame_copy, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Add hands detected text
        if hands_detected:
            hands_text = f"Hands: {', '.join(hands_detected)}"
            cv2.putText(frame_copy, hands_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame_copy