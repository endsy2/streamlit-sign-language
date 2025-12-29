"""
Webcam tab for real-time sign language recognition.
Collects sequence silently, then shows prediction.
"""

import streamlit as st
import cv2
import numpy as np
from collections import deque
from keypoint_extractor import KeypointExtractor
import time

from ui_conponent import UIComponents


def run_webcam_tab(settings, model_handler):
    """Run the webcam tab with batch prediction after collection."""
    st.header("Real-time Webcam Detection")
    st.info(f"📊 Will predict after collecting **{settings['sequence_length']} frames**")

    # Initialize session state
    if 'collecting' not in st.session_state:
        st.session_state.collecting = True
        st.session_state.showing_result = False
        st.session_state.result_data = None
        st.session_state.sequence = deque(maxlen=settings['sequence_length'])
        st.session_state.last_sequence_length = settings['sequence_length']
        st.session_state.frames_with_hands = 0  # Track frames with detected hands
        st.session_state.total_frames_collected = 0  # Track total frames

    # Check if sequence length changed - reset if so
    if st.session_state.get('last_sequence_length') != settings['sequence_length']:
        st.session_state.collecting = True
        st.session_state.showing_result = False
        st.session_state.result_data = None
        st.session_state.sequence = deque(maxlen=settings['sequence_length'])
        st.session_state.last_sequence_length = settings['sequence_length']
        st.session_state.frames_with_hands = 0
        st.session_state.total_frames_collected = 0
        st.info(f"⚙️ Sequence length changed to {settings['sequence_length']}. Restarting collection...")

    col1, col2 = st.columns([2, 1])

    with col1:
        run_webcam = st.checkbox("Start Webcam")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

    with col2:
        result_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        action_placeholder = st.empty()

    # Reset button (only show when showing results)
    if st.session_state.showing_result:
        with action_placeholder.container():
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 **Next Prediction**", type="primary", use_container_width=True):
                    # Reset everything
                    st.session_state.collecting = True
                    st.session_state.showing_result = False
                    st.session_state.result_data = None
                    st.session_state.sequence.clear()
                    st.session_state.frames_with_hands = 0  # Reset counter
                    st.session_state.total_frames_collected = 0  # Reset counter
                    st.rerun()

    if run_webcam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            return

        extractor = KeypointExtractor(
            min_detection_confidence=settings['min_detection_conf'],
            min_tracking_confidence=settings['min_tracking_conf']
        )

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

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # STATE 1: COLLECTING DATA
            if st.session_state.collecting:
                st.session_state.sequence.append(keypoints)
                st.session_state.total_frames_collected += 1

                # Track if hands were detected in this frame
                if hands:
                    st.session_state.frames_with_hands += 1

                # Add collection status overlay
                overlay_frame = rgb_frame.copy()

                # Check if hands are detected
                if not hands:
                    # NO HANDS DETECTED - Show warning
                    cv2.putText(
                        overlay_frame,
                        "⚠ NO HANDS DETECTED!",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),  # Red
                        3
                    )
                    cv2.putText(
                        overlay_frame,
                        "Please show your hand to camera",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),  # Red
                        2
                    )
                else:
                    # HANDS DETECTED - Show normal collecting status
                    cv2.putText(
                        overlay_frame,
                        f"Collecting: {len(st.session_state.sequence)}/{settings['sequence_length']}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 0),  # Yellow
                        3
                    )

                    # Show hands detected
                    cv2.putText(
                        overlay_frame,
                        f"Hands: {', '.join(hands)}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                frame_placeholder.image(overlay_frame, channels="RGB", use_container_width=True)

                # Calculate percentage of frames with hands
                hand_detection_rate = (st.session_state.frames_with_hands / st.session_state.total_frames_collected * 100) if st.session_state.total_frames_collected > 0 else 0

                # Progress bar
                with progress_placeholder.container():
                    progress = len(st.session_state.sequence) / settings['sequence_length']
                    st.progress(progress, text=f"Collecting: {len(st.session_state.sequence)}/{settings['sequence_length']} frames | Hands detected: {hand_detection_rate:.0f}%")

                # Status message based on hand detection
                with status_placeholder.container():
                    if not hands:
                        st.error("❌ No hands detected! Please show your hand to the camera")
                    else:
                        st.info(f"🎥 Collecting frames... Hold your gesture steady | Hands: {', '.join(hands)}")

                # Check if collection complete
                if len(st.session_state.sequence) == settings['sequence_length']:
                    # Calculate minimum required frames with hands (70% threshold)
                    min_required_frames = settings['sequence_length'] * 0.7

                    if st.session_state.frames_with_hands < min_required_frames:
                        # NOT ENOUGH HANDS DETECTED - Show error instead of prediction
                        st.session_state.result_data = {
                            'error': True,
                            'frames_with_hands': st.session_state.frames_with_hands,
                            'total_frames': st.session_state.total_frames_collected,
                            'detection_rate': hand_detection_rate
                        }
                    else:
                        # ENOUGH HANDS DETECTED - Make prediction
                        label, conf, preds = model_handler.predict(
                            st.session_state.sequence, settings['sequence_length']
                        )

                        # Store result
                        st.session_state.result_data = {
                            'error': False,
                            'label': label,
                            'confidence': conf,
                            'predictions': preds,
                            'hands': hands,
                            'frames_with_hands': st.session_state.frames_with_hands,
                            'total_frames': st.session_state.total_frames_collected,
                            'detection_rate': hand_detection_rate
                        }

                    # Switch to showing result
                    st.session_state.collecting = False
                    st.session_state.showing_result = True
                    st.rerun()  # Refresh to show results

            # STATE 2: SHOWING RESULT
            elif st.session_state.showing_result and st.session_state.result_data:

                # CHECK IF ERROR (Not enough hands detected)
                if st.session_state.result_data.get('error', False):
                    # SHOW ERROR SCREEN
                    overlay_frame = rgb_frame.copy()

                    # Big red X
                    cv2.putText(
                        overlay_frame,
                        "✗ COLLECTION FAILED",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),  # Red
                        3
                    )

                    cv2.putText(
                        overlay_frame,
                        "Not enough hands detected",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),  # Red
                        2
                    )

                    frame_placeholder.image(overlay_frame, channels="RGB", use_container_width=True)
                    progress_placeholder.empty()

                    # Show error message
                    with status_placeholder.container():
                        st.error("❌ Collection failed - Not enough hand keypoints detected!")

                    with result_placeholder.container():
                        st.error("### ⚠️ Cannot Make Prediction")
                        st.warning(f"""
                        **Insufficient Hand Detection**
                        
                        - Frames with hands detected: **{st.session_state.result_data['frames_with_hands']}/{st.session_state.result_data['total_frames']}** ({st.session_state.result_data['detection_rate']:.0f}%)
                        - Minimum required: **{int(st.session_state.result_data['total_frames'] * 0.7)}** frames (70%)
                        
                        **Please try again and:**
                        - Keep your hand clearly visible in the camera
                        - Ensure good lighting conditions
                        - Hold your gesture steady throughout collection
                        - Keep your entire hand within the frame
                        """)

                    # Clear other placeholders
                    metrics_placeholder.empty()
                    chart_placeholder.empty()

                else:
                    # SHOW SUCCESSFUL PREDICTION RESULT
                    overlay_frame = rgb_frame.copy()

                    # Color based on confidence
                    if st.session_state.result_data['confidence'] > settings['confidence_threshold']:
                        color = (0, 255, 0)  # Green
                        status = "✓"
                    else:
                        color = (255, 165, 0)  # Orange
                        status = "?"

                    cv2.putText(
                        overlay_frame,
                        f"{status} {st.session_state.result_data['label']} ({st.session_state.result_data['confidence']*100:.1f}%)",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        color,
                        3
                    )

                    frame_placeholder.image(overlay_frame, channels="RGB", use_container_width=True)
                    progress_placeholder.empty()

                    # Show status
                    with status_placeholder.container():
                        st.success(f"✅ Prediction complete! Hand detection: {st.session_state.result_data['detection_rate']:.0f}%")

                    # Show detailed results
                    with result_placeholder.container():
                        if st.session_state.result_data['confidence'] > settings['confidence_threshold']:
                            st.success(f"**Prediction:** {st.session_state.result_data['label']}")
                        elif st.session_state.result_data['confidence'] > settings['confidence_threshold'] * 0.6:
                            st.warning(f"**Prediction:** {st.session_state.result_data['label']}")
                        else:
                            st.info(f"**Prediction:** {st.session_state.result_data['label']} (Low confidence)")

                    with metrics_placeholder.container():
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Confidence", f"{st.session_state.result_data['confidence']:.2%}")
                        with col_b:
                            hands_text = ", ".join(st.session_state.result_data['hands']) if st.session_state.result_data['hands'] else "None"
                            st.metric("Hands", hands_text)
                        with col_c:
                            st.metric("Detection Rate", f"{st.session_state.result_data['detection_rate']:.0f}%")

                    # Top predictions chart
                    with chart_placeholder.container():
                        if st.session_state.result_data['predictions'].sum() > 0:
                            top_k = min(5, len(model_handler.class_labels))
                            top_k_idx = np.argsort(st.session_state.result_data['predictions'])[-top_k:][::-1]

                            # Safety check
                            top_k_idx = [i for i in top_k_idx if i < len(model_handler.class_labels)]

                            if top_k_idx:
                                top_k_labels = [model_handler.class_labels[i] for i in top_k_idx]
                                top_k_values = [float(st.session_state.result_data['predictions'][i]) for i in top_k_idx]

                                st.caption("Top 5 Predictions")
                                chart_data = {label: val for label, val in zip(top_k_labels, top_k_values)}
                                st.bar_chart(chart_data)

                # Break the loop to stop processing frames
                break

            if stop_button:
                break

        extractor.close()
        cap.release()