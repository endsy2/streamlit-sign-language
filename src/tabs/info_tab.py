"""
Info Tab for Sign Language Recognition
"""
import streamlit as st


def run_info_tab(settings, model_handler):
    """Run the information tab."""
    st.header("ℹ️ System Information")

    st.markdown(f"""
    ### Architecture

    **Input Pipeline:**
    1. MediaPipe detects hands (up to 2)
    2. Extracts 21 landmarks per hand in 3D
    3. Creates vector of 126 values per frame
    4. Collects sequence of {settings['sequence_length']} frames
    5. Feeds to model: shape (1, {settings['sequence_length']}, 126)

    **Model Configuration:**
    - Model Path: `{settings['model_path']}`
    - Classes: {len(model_handler.class_labels)}
    - Sequence Length: {settings['sequence_length']} frames

    ### Keypoint Structure

    **Per Hand (63 values):**
    - 21 landmarks × 3 coordinates (x, y, z)

    **Total: 126 values**
    - Left hand: 63 values
    - Right hand: 63 values

    ### Tips for Best Results
    - Ensure good lighting
    - Keep hands in frame
    - Wait for sequence to fill
    - Natural gesture speed works best
    """)
