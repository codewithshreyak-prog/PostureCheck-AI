import math


class PostureAnalyzer:
    """Analyze basic front-facing posture using pose landmarks."""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def analyze(self, landmarks) -> dict:
        """Return posture measurements, score, and feedback."""

        if not landmarks:
            return {
                "status": "No body detected",
                "score": 0,
                "shoulder_angle": 0,
                "head_tilt": 0,
                "feedback": ["Move backward and keep your upper body visible."],
            }

        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]

        left_ear = landmarks[self.LEFT_EAR]
        right_ear = landmarks[self.RIGHT_EAR]

        shoulder_angle = self._calculate_angle(
            left_shoulder.x,
            left_shoulder.y,
            right_shoulder.x,
            right_shoulder.y,
        )

        head_tilt = self._calculate_angle(
            left_ear.x,
            left_ear.y,
            right_ear.x,
            right_ear.y,
        )

        feedback = []
        score = 100

        # Shoulder alignment
        if abs(shoulder_angle) <= 4:
            feedback.append("Shoulders are well aligned.")
        elif abs(shoulder_angle) <= 8:
            feedback.append("Slight shoulder imbalance detected.")
            score -= 15
        else:
            feedback.append("Shoulders are noticeably uneven.")
            score -= 30

        # Head alignment
        if abs(head_tilt) <= 5:
            feedback.append("Head position looks centered.")
        elif abs(head_tilt) <= 10:
            feedback.append("Slight head tilt detected.")
            score -= 15
        else:
            feedback.append("Significant head tilt detected.")
            score -= 30

        score = max(score, 0)

        if score >= 85:
            status = "Good posture"
        elif score >= 60:
            status = "Moderate posture"
        else:
            status = "Poor posture"

        return {
            "status": status,
            "score": score,
            "shoulder_angle": shoulder_angle,
            "head_tilt": head_tilt,
            "feedback": feedback,
        }

    @staticmethod
    def _calculate_angle(x1, y1, x2, y2) -> float:
        """Calculate the angle of a line relative to the horizontal axis."""

        delta_y = y2 - y1
        delta_x = x2 - x1

        return math.degrees(math.atan2(delta_y, delta_x))