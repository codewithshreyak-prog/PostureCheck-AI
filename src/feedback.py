class PostureFeedbackGenerator:
    """Generate personalized posture correction recommendations."""

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def generate(
        self,
        analysis: dict,
        landmarks=None,
    ) -> list[str]:
        """Generate corrections based on posture measurements."""

        if not analysis.get("valid", False):
            return analysis.get(
                "feedback",
                ["Move into a clearer camera position."],
            )

        recommendations = []

        shoulder_angle = analysis["shoulder_angle"]
        head_tilt = analysis["head_tilt"]
        hip_angle = analysis["hip_angle"]
        torso_lean = analysis["torso_lean"]

        # Shoulder correction
        if abs(shoulder_angle) > 4:
            recommendation = self._shoulder_recommendation(
                landmarks
            )
            recommendations.append(recommendation)

        # Head correction
        if abs(head_tilt) > 5:
            recommendations.append(
                "Level your head and look straight ahead."
            )

        # Hip correction
        if abs(hip_angle) > 4:
            recommendations.append(
                "Distribute your weight evenly and level your hips."
            )

        # Torso correction
        if abs(torso_lean) > 4:
            recommendations.append(
                "Center your shoulders directly above your hips."
            )

        if not recommendations:
            recommendations.append(
                "Great alignment. Maintain this posture."
            )

        return recommendations[:3]

    def _shoulder_recommendation(
        self,
        landmarks,
    ) -> str:
        """Identify the lower shoulder and generate a correction."""

        if not landmarks:
            return "Relax and level both shoulders."

        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]

        # A larger y-coordinate means the landmark is lower.
        if left_shoulder.y > right_shoulder.y:
            return "Raise your left shoulder slightly."

        if right_shoulder.y > left_shoulder.y:
            return "Raise your right shoulder slightly."

        return "Relax and level both shoulders."