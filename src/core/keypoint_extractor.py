"""
Keypoint Extractor using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple


class KeypointExtractor:
    """Extract hand keypoints using MediaPipe."""

    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 max_num_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.num_landmarks = 21

    def extract(self, frame: np.ndarray, show_keypoints: bool = True) -> Tuple[List[float], np.ndarray, List[str]]:
        """Extract keypoints from frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_hand = [(0.0, 0.0, 0.0)] * self.num_landmarks
        right_hand = [(0.0, 0.0, 0.0)] * self.num_landmarks
        annotated_frame = frame.copy()
        hands_detected = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                kp = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                if label == "Left":
                    left_hand = kp
                    hands_detected.append("Left")
                else:
                    right_hand = kp
                    hands_detected.append("Right")

                if show_keypoints:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

        keypoints = [coord for kp in left_hand + right_hand for coord in kp]
        return keypoints, annotated_frame, hands_detected

    def close(self):
        """Release resources."""
        self.hands.close()
