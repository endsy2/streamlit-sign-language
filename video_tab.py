
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
import os
from collections import deque, Counter
import tempfile
# ============================================================================
# VIDEO TAB
# ============================================================================
def run_video_tab(settings, model_handler):
    """Run the video upload tab."""
    st.header("Video File Analysis")

    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.video(uploaded_video)

        with col2:
            process_btn = st.button("🔍 Process Video")
            frame_skip = st.slider("Process every N frames", 1, 5, 1)

        if process_btn:
            st.write("Processing video...")

            extractor = KeypointExtractor(
                min_detection_confidence=settings['min_detection_conf'],
                min_tracking_confidence=settings['min_tracking_conf']
            )
            sequence = deque(maxlen=settings['sequence_length'])

            progress_bar = st.progress(0)
            status_text = st.empty()

            predictions_list = []

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.flip(frame, 1)
                    keypoints, _, hands = extractor.extract(frame, False)
                    sequence.append(keypoints)

                    if len(sequence) == settings['sequence_length']:
                        label, conf, _ = model_handler.predict(
                            sequence, settings['sequence_length']
                        )
                        predictions_list.append({
                            'frame': frame_count,
                            'label': label,
                            'confidence': conf,
                            'hands': len(hands)
                        })

                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

                if frame_count % 30 == 0:
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

            cap.release()
            extractor.close()

            st.success("✅ Processing complete!")

            if predictions_list:
                st.subheader("Analysis Results")

                valid_preds = [
                    p for p in predictions_list
                    if p['confidence'] > settings['confidence_threshold']
                ]

                if valid_preds:
                    counts = Counter([p['label'] for p in valid_preds])

                    st.bar_chart(counts)

                    st.write("**Most Common Signs:**")
                    for label, count in counts.most_common(5):
                        pct = count / len(valid_preds) * 100
                        st.write(f"- {label}: {count} times ({pct:.1f}%)")

                    avg_conf = np.mean([p['confidence'] for p in valid_preds])
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                else:
                    st.warning("No predictions above threshold")

        if os.path.exists(tfile.name):
            os.unlink(tfile.name)