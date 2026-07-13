import cv2

from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer
from src.posture_smoother import PostureSmoother


def draw_posture_information(
    frame,
    analysis,
) -> None:
    """Display posture measurements and guidance."""

    is_valid = analysis.get(
        "valid",
        False,
    )

    score = analysis["score"]

    if not is_valid:
        posture_color = (0, 165, 255)
    elif score >= 85:
        posture_color = (0, 255, 0)
    elif score >= 60:
        posture_color = (0, 255, 255)
    else:
        posture_color = (0, 0, 255)

    if analysis.get("is_stable", False):
        stability_color = (0, 255, 0)
    else:
        stability_color = (0, 165, 255)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (590, 340),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame,
    )

    cv2.putText(
        frame,
        f"Status: {analysis['status']}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        posture_color,
        2,
    )

    if is_valid:
        score_text = (
            f"Posture Score: {score}/100"
        )
    else:
        score_text = (
            "Posture Score: Not available"
        )

    cv2.putText(
        frame,
        score_text,
        (25, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        posture_color,
        2,
    )

    cv2.putText(
        frame,
        analysis.get(
            "stability_message",
            "Waiting for posture data",
        ),
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        stability_color,
        2,
    )

    y_position = 145

    if is_valid:
        measurements = [
            (
                f"Shoulder angle: "
                f"{analysis['shoulder_angle']:.1f} deg"
            ),
            (
                f"Head tilt: "
                f"{analysis['head_tilt']:.1f} deg"
            ),
            (
                f"Hip angle: "
                f"{analysis['hip_angle']:.1f} deg"
            ),
            (
                f"Torso lean: "
                f"{analysis['torso_lean']:.1f} deg"
            ),
        ]

        for measurement in measurements:
            cv2.putText(
                frame,
                measurement,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 255),
                1,
            )

            y_position += 28

    for feedback in analysis["feedback"][:3]:
        cv2.putText(
            frame,
            feedback,
            (25, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            posture_color,
            1,
        )

        y_position += 26


def start_webcam() -> None:
    """Start the posture detection application."""

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Unable to access the MacBook camera."
        )

    pose_detector = PoseDetector()
    posture_analyzer = PostureAnalyzer()

    posture_smoother = PostureSmoother(
        window_size=12,
        minimum_frames=6,
    )

    print("PostureCheck AI started.")
    print(
        "Keep your head, shoulders, and hips visible."
    )
    print(
        "Hold still briefly while posture is calibrated."
    )
    print("Press Q to close the application.")

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print(
                    "Unable to read webcam frame."
                )
                continue

            frame = cv2.flip(
                frame,
                1,
            )

            frame, pose_results = (
                pose_detector.detect_pose(
                    frame
                )
            )

            if pose_results.pose_landmarks:
                landmarks = (
                    pose_results.pose_landmarks[0]
                )

                raw_analysis = (
                    posture_analyzer.analyze(
                        landmarks
                    )
                )
            else:
                raw_analysis = (
                    posture_analyzer.empty_analysis()
                )

            analysis = posture_smoother.update(
                raw_analysis
            )

            draw_posture_information(
                frame,
                analysis,
            )

            cv2.imshow(
                "PostureCheck AI",
                frame,
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:
        camera.release()
        pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()