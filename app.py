import cv2

from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer


def draw_posture_information(frame, analysis):
    """Display posture measurements and feedback."""

    score = analysis["score"]

    if score >= 85:
        color = (0, 255, 0)
    elif score >= 60:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (525, 275),
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
        color,
        2,
    )

    cv2.putText(
        frame,
        f"Posture Score: {score}/100",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
    )

    measurements = [
        f"Shoulder angle: {analysis['shoulder_angle']:.1f} deg",
        f"Head tilt: {analysis['head_tilt']:.1f} deg",
        f"Hip angle: {analysis['hip_angle']:.1f} deg",
        f"Torso lean: {analysis['torso_lean']:.1f} deg",
    ]

    y_position = 108

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

        y_position += 27

    for feedback in analysis["feedback"][:2]:
        cv2.putText(
            frame,
            feedback,
            (25, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )

        y_position += 25


def start_webcam() -> None:
    """Start the posture detection webcam application."""

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

    print("PostureCheck AI started.")
    print("Keep your head, shoulders, and hips visible.")
    print("Press Q to close the application.")

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Unable to read webcam frame.")
                continue

            frame = cv2.flip(frame, 1)

            frame, pose_results = pose_detector.detect_pose(frame)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks[0]
                analysis = posture_analyzer.analyze(landmarks)
            else:
                analysis = posture_analyzer.empty_analysis()

            draw_posture_information(
                frame,
                analysis,
            )

            cv2.imshow(
                "PostureCheck AI",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()