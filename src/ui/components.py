"""
UI Components for Sign Language Recognition
"""
import streamlit as st
import cv2
import numpy as np
from typing import List


class UIComponents:
    """Reusable UI components."""

    @staticmethod
    def display_prediction(label: str, confidence: float, threshold: float = 0.3):
        """Display prediction with styling - always show result."""
        st.markdown(f"### 🎯 Result: **{label}**")
        st.markdown(f"**Confidence: {confidence:.1%}**")
        
        if confidence > threshold:
            st.success(f"✅ High confidence prediction")
        elif confidence > 0.15:
            st.warning(f"⚠️ Medium confidence - try again for better result")
        else:
            st.info(f"🔍 Low confidence - please try again")

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
    def display_top_predictions(predictions: np.ndarray, class_labels: List[str], top_k: int = 5):
        """Display top predictions with percentages."""
        if predictions.sum() == 0:
            st.warning("No predictions available")
            return

        top_k = min(top_k, len(class_labels))
        top_k_idx = np.argsort(predictions)[-top_k:][::-1]
        
        st.markdown("#### Top 5 Predictions:")
        for i, idx in enumerate(top_k_idx):
            label = class_labels[idx]
            conf = predictions[idx]
            bar = "🟩" * int(conf * 10) + "⬜" * (10 - int(conf * 10))
            st.markdown(f"{i+1}. **{label}**: {conf:.1%} {bar}")

    @staticmethod
    def display_sequence_progress(current: int, total: int):
        """Display sequence collection progress bar."""
        progress = min(current / total, 1.0)
        st.progress(progress, text=f"Collecting: {current}/{total} frames")

    @staticmethod
    def display_frame_with_overlay(frame: np.ndarray, label: str, confidence: float,
                                   count: int, total: int, hands: List[str], threshold: float = 0.5):
        """Add text overlays to frame."""
        frame_copy = frame.copy()

        if count >= total:
            color = (0, 255, 0) if confidence > threshold else (255, 165, 0)
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.putText(frame_copy, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(frame_copy, f"Collecting: {count}/{total}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        if hands:
            cv2.putText(frame_copy, f"Hands: {', '.join(hands)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame_copy
