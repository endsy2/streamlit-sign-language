"""
Webcam Tab for Sign Language Recognition
"""
import streamlit as st
import cv2
import numpy as np
from collections import deque
from src.core.keypoint_extractor import KeypointExtractor
from src.ui.components import UIComponents


def run_webcam_tab(settings, model_handler):
    """Run the webcam tab."""
    st.header("Real-time Webcam Detection")
    st.info(f"📊 Collecting **{settings['sequence_length']} frames** with hand keypoints")

    # Initialize session state
    if 'sequence' not in st.session_state or st.session_state.get('seq_length') != settings['sequence_length']:
        st.session_state.sequence = deque(maxlen=settings['sequence_length'])
        st.session_state.seq_length = settings['sequence_length']
        st.session_state.prediction_done = False
        st.session_state.last_prediction = None
        st.session_state.last_frame = None
        st.session_state.frames_with_hands = 0
    
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'frames_with_hands' not in st.session_state:
        st.session_state.frames_with_hands = 0

    col1, col2 = st.columns([2, 1])

    with col1:
        run_webcam = st.checkbox("Start Webcam")
        frame_placeholder = st.empty()
        progress_placeholder = st.empty()
        alert_placeholder = st.empty()

    with col2:
        prediction_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        button_placeholder = st.empty()

    # Show Re-predict button when prediction is done
    if st.session_state.prediction_done:
        with button_placeholder.container():
            if st.button("🔄 Re-predict", type="primary", use_container_width=True):
                st.session_state.sequence.clear()
                st.session_state.prediction_done = False
                st.session_state.last_prediction = None
                st.session_state.last_frame = None
                st.session_state.frames_with_hands = 0
                st.rerun()
        
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
        
        with progress_placeholder.container():
            st.success("✅ Prediction complete! Click **Re-predict** to try again.")
        
        if st.session_state.last_prediction:
            ui = UIComponents()
            with prediction_placeholder.container():
                ui.display_prediction(
                    st.session_state.last_prediction['label'], 
                    st.session_state.last_prediction['confidence'], 
                    settings['confidence_threshold']
                )
            with metrics_placeholder.container():
                ui.display_confidence_metrics(
                    st.session_state.last_prediction['confidence'], 
                    st.session_state.last_prediction['hands']
                )
            with chart_placeholder.container():
                ui.display_top_predictions(
                    st.session_state.last_prediction['predictions'], 
                    model_handler.class_labels
                )
        return

    if run_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            return

        # Optimized camera settings for smooth performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        extractor = KeypointExtractor(
            min_detection_confidence=settings['min_detection_conf'],
            min_tracking_confidence=settings['min_tracking_conf']
        )

        ui = UIComponents()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            
            keypoints, annotated_frame, hands = extractor.extract(frame, settings['show_keypoints'])
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            if hands:
                st.session_state.sequence.append(keypoints)
                st.session_state.frames_with_hands += 1
                
                with alert_placeholder.container():
                    st.success(f"✅ Hand detected: {', '.join(hands)}")
                
                with progress_placeholder.container():
                    progress = len(st.session_state.sequence) / settings['sequence_length']
                    st.progress(progress, text=f"Collecting: {len(st.session_state.sequence)}/{settings['sequence_length']} frames")

                with prediction_placeholder.container():
                    st.info(f"📹 Recording... {len(st.session_state.sequence)}/{settings['sequence_length']}")
            else:
                with alert_placeholder.container():
                    st.error("⚠️ **NO HAND DETECTED!** Please show your hand to the camera.")
                
                with progress_placeholder.container():
                    progress = len(st.session_state.sequence) / settings['sequence_length']
                    st.progress(progress, text=f"Waiting... {len(st.session_state.sequence)}/{settings['sequence_length']} frames")

                with prediction_placeholder.container():
                    st.warning("👋 Show your hand to collect keypoints")

            if len(st.session_state.sequence) == settings['sequence_length']:
                label, conf, preds = model_handler.predict(st.session_state.sequence, settings['sequence_length'])
                
                st.session_state.prediction_done = True
                st.session_state.last_prediction = {
                    'label': label, 'confidence': conf, 'predictions': preds, 'hands': hands
                }
                st.session_state.last_frame = rgb_frame
                
                extractor.close()
                cap.release()
                st.rerun()
                return

        extractor.close()
        cap.release()
