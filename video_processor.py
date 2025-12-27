"""
Process video files and webcam streams.
Handles video I/O and frame processing.
"""

import cv2
import numpy as np
from collections import deque
from typing import Callable, List, Dict, Any


class VideoProcessor:
    def __init__(self, sequence_length: int):
        """
        Initialize video processor.

        Args:
            sequence_length: Number of frames in sequence buffer
        """
        self.sequence_length = sequence_length
        self.sequence = deque(maxlen=sequence_length)

    def reset_sequence(self):
        """Clear the sequence buffer."""
        self.sequence.clear()

    def add_to_sequence(self, keypoints: List[float]):
        """
        Add keypoints to sequence buffer.

        Args:
            keypoints: List of 126 keypoint values
        """
        self.sequence.append(keypoints)

    def is_sequence_ready(self) -> bool:
        """Check if sequence buffer is full."""
        return len(self.sequence) == self.sequence_length

    def get_sequence_progress(self) -> float:
        """Get sequence collection progress (0.0 to 1.0)."""
        return len(self.sequence) / self.sequence_length

    def process_video_file(self,
                           video_path: str,
                           keypoint_extractor: 'KeypointExtractor',
                           predictor: Callable,
                           frame_skip: int = 1,
                           flip_horizontal: bool = True) -> List[Dict[str, Any]]:
        """
        Process entire video file.

        Args:
            video_path: Path to video file
            keypoint_extractor: KeypointExtractor instance
            predictor: Function that takes sequence and returns prediction
            frame_skip: Process every N frames
            flip_horizontal: Whether to flip frames horizontally

        Returns:
            List of prediction results
        """
        cap = cv2.VideoCapture(video_path)
        predictions = []
        frame_count = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        self.reset_sequence()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Flip frame if needed
                if flip_horizontal:
                    frame = cv2.flip(frame, 1)

                # Extract keypoints
                keypoints, _, hands_detected = keypoint_extractor.extract(
                    frame, show_keypoints=False
                )

                # Add to sequence
                self.add_to_sequence(keypoints)

                # Make prediction if sequence is ready
                if self.is_sequence_ready():
                    result = predictor(self.sequence)
                    predictions.append({
                        'frame': frame_count,
                        'result': result,
                        'hands': len(hands_detected)
                    })

            frame_count += 1

            # Yield progress for UI updates
            if frame_count % 10 == 0:
                yield {
                    'progress': frame_count / total_frames,
                    'frame': frame_count,
                    'total': total_frames
                }

        cap.release()

        yield {
            'progress': 1.0,
            'predictions': predictions,
            'fps': fps,
            'total_frames': total_frames
        }