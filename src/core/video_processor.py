"""
Video Processor for Sign Language Recognition
"""
import cv2
from collections import deque
from typing import Callable, List, Dict, Any


class VideoProcessor:
    """Process video files and webcam streams."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.sequence = deque(maxlen=sequence_length)

    def reset_sequence(self):
        """Clear the sequence buffer."""
        self.sequence.clear()

    def add_to_sequence(self, keypoints: List[float]):
        """Add keypoints to sequence buffer."""
        self.sequence.append(keypoints)

    def is_sequence_ready(self) -> bool:
        """Check if sequence buffer is full."""
        return len(self.sequence) == self.sequence_length

    def get_progress(self) -> float:
        """Get sequence collection progress (0.0 to 1.0)."""
        return len(self.sequence) / self.sequence_length

    def process_video(self, video_path: str, keypoint_extractor, predictor: Callable,
                      frame_skip: int = 1) -> List[Dict[str, Any]]:
        """Process entire video file."""
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
                frame = cv2.flip(frame, 1)
                keypoints, _, hands = keypoint_extractor.extract(frame, False)
                self.add_to_sequence(keypoints)

                if self.is_sequence_ready():
                    result = predictor(self.sequence)
                    predictions.append({'frame': frame_count, 'result': result, 'hands': len(hands)})

            frame_count += 1
            if frame_count % 10 == 0:
                yield {'progress': frame_count / total_frames, 'frame': frame_count}

        cap.release()
        yield {'progress': 1.0, 'predictions': predictions, 'fps': fps, 'total': total_frames}
