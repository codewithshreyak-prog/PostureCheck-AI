import math


class PostureAnalyzer:
    """Analyze front-facing posture using MediaPipe pose landmarks."""

    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    MIN_VISIBILITY = 0.55

    LANDMARK_NAMES = {
        LEFT_EAR: "left ear",
        RIGHT_EAR: "right ear",
        LEFT_SHOULDER: "left shoulder",
        RIGHT_SHOULDER: "right shoulder",
        LEFT_HIP: "left hip",
        RIGHT_HIP: "right hip",
    }

    REQUIRED_LANDMARKS = (
        LEFT_EAR,
        RIGHT_EAR,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    )

    def analyze(self, landmarks) -> dict:
        """Calculate posture measurements, score, and feedback."""

        if not landmarks:
            return self.invalid_analysis(
                status="No body detected",
                message="Move into the camera frame.",
            )

        missing_landmarks = self._find_missing_landmarks(
            landmarks
        )

        if missing_landmarks:
            return self._camera_guidance(
                missing_landmarks
            )

        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]

        left_ear = landmarks[self.LEFT_EAR]
        right_ear = landmarks[self.RIGHT_EAR]

        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        shoulder_angle = self._horizontal_angle(
            left_shoulder,
            right_shoulder,
        )

        head_tilt = self._horizontal_angle(
            left_ear,
            right_ear,
        )

        hip_angle = self._horizontal_angle(
            left_hip,
            right_hip,
        )

        shoulder_center = self._midpoint(
            left_shoulder,
            right_shoulder,
        )

        hip_center = self._midpoint(
            left_hip,
            right_hip,
        )

        torso_lean = self._vertical_angle(
            shoulder_center,
            hip_center,
        )

        score = 100
        feedback = []

        # Shoulder alignment
        if abs(shoulder_angle) <= 4:
            feedback.append("Shoulders are aligned.")
        elif abs(shoulder_angle) <= 8:
            feedback.append(
                "Slight shoulder imbalance detected."
            )
            score -= 10
        else:
            feedback.append(
                "Shoulders are noticeably uneven."
            )
            score -= 20

        # Head alignment
        if abs(head_tilt) <= 5:
            feedback.append(
                "Head position is centered."
            )
        elif abs(head_tilt) <= 10:
            feedback.append(
                "Slight head tilt detected."
            )
            score -= 10
        else:
            feedback.append(
                "Significant head tilt detected."
            )
            score -= 20

        # Hip alignment
        if abs(hip_angle) <= 4:
            feedback.append("Hips are aligned.")
        elif abs(hip_angle) <= 8:
            feedback.append(
                "Slight hip imbalance detected."
            )
            score -= 10
        else:
            feedback.append(
                "Hips are noticeably uneven."
            )
            score -= 20

        # Torso alignment
        if abs(torso_lean) <= 4:
            feedback.append("Torso is upright.")
        elif abs(torso_lean) <= 8:
            direction = (
                "right"
                if torso_lean > 0
                else "left"
            )

            feedback.append(
                f"Torso is leaning slightly {direction}."
            )
            score -= 10
        else:
            direction = (
                "right"
                if torso_lean > 0
                else "left"
            )

            feedback.append(
                f"Torso is leaning significantly {direction}."
            )
            score -= 20

        score = max(score, 0)

        if score >= 85:
            status = "Good posture"
        elif score >= 60:
            status = "Moderate posture"
        else:
            status = "Poor posture"

        return {
            "valid": True,
            "status": status,
            "score": score,
            "shoulder_angle": shoulder_angle,
            "head_tilt": head_tilt,
            "hip_angle": hip_angle,
            "torso_lean": torso_lean,
            "feedback": feedback,
            "missing_landmarks": [],
        }

    def _find_missing_landmarks(
        self,
        landmarks,
    ) -> list[str]:
        """Return required landmarks with low visibility."""

        missing = []

        for landmark_index in self.REQUIRED_LANDMARKS:
            landmark = landmarks[landmark_index]

            visibility = self._get_visibility(
                landmark
            )

            if visibility < self.MIN_VISIBILITY:
                missing.append(
                    self.LANDMARK_NAMES[landmark_index]
                )

        return missing

    def _camera_guidance(
        self,
        missing_landmarks: list[str],
    ) -> dict:
        """Generate camera-position guidance."""

        missing_text = " and ".join(
            missing_landmarks
        )

        hips_missing = any(
            "hip" in name
            for name in missing_landmarks
        )

        shoulders_missing = any(
            "shoulder" in name
            for name in missing_landmarks
        )

        ears_missing = any(
            "ear" in name
            for name in missing_landmarks
        )

        if hips_missing:
            message = (
                "Move farther back until both hips "
                "are clearly visible."
            )
        elif shoulders_missing:
            message = (
                "Keep both shoulders inside the frame."
            )
        elif ears_missing:
            message = (
                "Face the camera and keep your full "
                "head visible."
            )
        else:
            message = (
                "Adjust your position for a clearer "
                "body view."
            )

        return self.invalid_analysis(
            status="Adjust camera",
            message=message,
            missing_landmarks=missing_text,
        )

    @staticmethod
    def _get_visibility(landmark) -> float:
        """Safely obtain landmark visibility."""

        visibility = getattr(
            landmark,
            "visibility",
            0.0,
        )

        if visibility is None:
            return 0.0

        return float(visibility)

    @staticmethod
    def _horizontal_angle(point1, point2) -> float:
        """Calculate normalized horizontal tilt."""

        delta_y = point2.y - point1.y
        delta_x = point2.x - point1.x

        angle = math.degrees(
            math.atan2(delta_y, delta_x)
        )

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        return angle

    @staticmethod
    def _midpoint(
        point1,
        point2,
    ) -> tuple[float, float]:
        """Return midpoint between two landmarks."""

        midpoint_x = (
            point1.x + point2.x
        ) / 2

        midpoint_y = (
            point1.y + point2.y
        ) / 2

        return midpoint_x, midpoint_y

    @staticmethod
    def _vertical_angle(
        upper_point: tuple[float, float],
        lower_point: tuple[float, float],
    ) -> float:
        """Calculate torso lean relative to vertical."""

        upper_x, upper_y = upper_point
        lower_x, lower_y = lower_point

        horizontal_difference = (
            upper_x - lower_x
        )

        vertical_difference = (
            lower_y - upper_y
        )

        return math.degrees(
            math.atan2(
                horizontal_difference,
                vertical_difference,
            )
        )

    @staticmethod
    def invalid_analysis(
        status: str,
        message: str,
        missing_landmarks: str = "",
    ) -> dict:
        """Return values when posture cannot be scored."""

        feedback = [message]

        if missing_landmarks:
            feedback.append(
                f"Low visibility: {missing_landmarks}"
            )

        return {
            "valid": False,
            "status": status,
            "score": 0,
            "shoulder_angle": 0.0,
            "head_tilt": 0.0,
            "hip_angle": 0.0,
            "torso_lean": 0.0,
            "feedback": feedback,
            "missing_landmarks": missing_landmarks,
        }

    @staticmethod
    def empty_analysis() -> dict:
        """Return values when no body is detected."""

        return PostureAnalyzer.invalid_analysis(
            status="No body detected",
            message=(
                "Move backward and keep your "
                "upper body visible."
            ),
        )