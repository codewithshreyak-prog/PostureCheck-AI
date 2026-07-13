import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    """Detect and draw body landmarks using MediaPipe Pose Landmarker."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "pose_landmarker_full.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at: {model_path}"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

    def detect_pose(self, frame):
        """Detect pose landmarks in a webcam frame."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1,
        )

        self.last_timestamp_ms = timestamp_ms

        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if results.pose_landmarks:
            self._draw_landmarks(
                frame,
                results.pose_landmarks[0],
            )

        return frame, results

    @staticmethod
    def _draw_landmarks(frame, landmarks) -> None:
        """Draw pose joints and skeleton connections."""

        frame_height, frame_width = frame.shape[:2]
        points = []

        for landmark in landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            visibility = getattr(landmark, "visibility", 1.0)

            if visibility is None:
                visibility = 1.0

            points.append((x, y, visibility))

        for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            start_index = connection.start
            end_index = connection.end

            start_x, start_y, start_visibility = points[start_index]
            end_x, end_y, end_visibility = points[end_index]

            if start_visibility > 0.5 and end_visibility > 0.5:
                cv2.line(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (0, 255, 0),
                    2,
                )

        for x, y, visibility in points:
            if visibility > 0.5:
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1,
                )

    def close(self) -> None:
        """Release MediaPipe resources."""

        self.landmarker.close()