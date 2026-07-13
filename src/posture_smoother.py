from collections import deque
from statistics import fmean, pstdev


class PostureSmoother:
    """Smooth posture measurements across video frames."""

    MEASUREMENT_KEYS = (
        "score",
        "shoulder_angle",
        "head_tilt",
        "hip_angle",
        "torso_lean",
    )

    def __init__(
        self,
        window_size: int = 12,
        minimum_frames: int = 6,
    ) -> None:
        self.window_size = window_size
        self.minimum_frames = minimum_frames

        self.history = {
            key: deque(maxlen=window_size)
            for key in self.MEASUREMENT_KEYS
        }

    def update(self, analysis: dict) -> dict:
        """Add a result and return smoothed posture data."""

        if not analysis.get("valid", False):
            self.reset()

            return {
                **analysis,
                "is_stable": False,
                "stability_message": (
                    "Waiting for a clear posture view"
                ),
            }

        for key in self.MEASUREMENT_KEYS:
            self.history[key].append(
                float(analysis[key])
            )

        smoothed = dict(analysis)

        smoothed["score"] = int(
            round(
                fmean(
                    self.history["score"]
                )
            )
        )

        for key in (
            "shoulder_angle",
            "head_tilt",
            "hip_angle",
            "torso_lean",
        ):
            smoothed[key] = fmean(
                self.history[key]
            )

        smoothed["status"] = self._classify_score(
            smoothed["score"]
        )

        current_frames = len(
            self.history["score"]
        )

        if current_frames < self.minimum_frames:
            smoothed["is_stable"] = False

            smoothed["stability_message"] = (
                f"Calibrating posture: "
                f"{current_frames}/"
                f"{self.minimum_frames}"
            )

            return smoothed

        score_variation = pstdev(
            self.history["score"]
        )

        angle_variations = [
            pstdev(
                self.history[key]
            )
            for key in (
                "shoulder_angle",
                "head_tilt",
                "hip_angle",
                "torso_lean",
            )
        ]

        maximum_angle_variation = max(
            angle_variations
        )

        is_stable = (
            score_variation <= 6.0
            and maximum_angle_variation <= 2.5
        )

        smoothed["is_stable"] = is_stable

        if is_stable:
            smoothed["stability_message"] = (
                "Stable posture reading"
            )
        else:
            smoothed["stability_message"] = (
                "Hold still for a stable reading"
            )

        return smoothed

    def reset(self) -> None:
        """Clear all stored measurements."""

        for values in self.history.values():
            values.clear()

    @staticmethod
    def _classify_score(score: int) -> str:
        """Convert posture score into a status."""

        if score >= 85:
            return "Good posture"

        if score >= 60:
            return "Moderate posture"

        return "Poor posture"