
"""
Sign Language Recognition System
Complete working version with proper import order
"""

# ============================================================================
# IMPORTS - MUST BE FIRST
# ============================================================================
import streamlit as st
import cv2
from collections import deque
from keypoint_extractor import KeypointExtractor
from ui_conponent import UIComponents



def run_webcam_tab(settings, model_handler):
    """Run the webcam tab."""
    st.header("Real-time Webcam Detection")
    st.info(f"📊 Collecting **{settings['sequence_length']} frames** before prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        run_webcam = st.checkbox("Start Webcam")
        frame_placeholder = st.empty()
        progress_placeholder = st.empty()

    with col2:
        prediction_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            return

        extractor = KeypointExtractor(
            min_detection_confidence=settings['min_detection_conf'],
            min_tracking_confidence=settings['min_tracking_conf']
        )

        sequence = deque(maxlen=settings['sequence_length'])
        ui = UIComponents()

        stop_button = st.button("Stop Webcam")

        while not stop_button:
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            keypoints, annotated_frame, hands = extractor.extract(
                frame, settings['show_keypoints']
            )

            sequence.append(keypoints)
            label, conf, preds = model_handler.predict(
                sequence, settings['sequence_length']
            )

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            display_frame = ui.display_frame_with_overlay(
                rgb_frame, label, conf,
                len(sequence), settings['sequence_length'],
                hands, settings['confidence_threshold']
            )

            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            with progress_placeholder.container():
                ui.display_sequence_progress(len(sequence), settings['sequence_length'])

            with prediction_placeholder.container():
                if len(sequence) == settings['sequence_length']:
                    ui.display_prediction(label, conf, settings['confidence_threshold'])
                else:
                    st.info(f"Collecting frames... ({len(sequence)}/{settings['sequence_length']})")

            with metrics_placeholder.container():
                if len(sequence) == settings['sequence_length']:
                    ui.display_confidence_metrics(conf, hands)

            with chart_placeholder.container():
                if len(sequence) == settings['sequence_length']:
                    ui.display_top_predictions(preds, model_handler.class_labels)

            if stop_button:
                break

        extractor.close()
        cap.release()