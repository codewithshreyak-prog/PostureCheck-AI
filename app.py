import cv2

from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer


def draw_posture_information(frame, analysis):
    """Display posture results on the webcam frame."""

    status = analysis["status"]
    score = analysis["score"]

    if score >= 85:
        color = (0, 255, 0)
    elif score >= 60:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (10, 10),
        (490, 185),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    cv2.putText(
        frame,
        f"Posture Score: {score}/100",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        frame,
        f"Shoulder angle: {analysis['shoulder_angle']:.1f} deg",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Head tilt: {analysis['head_tilt']:.1f} deg",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if analysis["feedback"]:
        cv2.putText(
            frame,
            analysis["feedback"][0],
            (25, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def start_webcam() -> None:
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError("Unable to access the MacBook camera.")

    pose_detector = PoseDetector()
    posture_analyzer = PostureAnalyzer()

    print("PostureCheck AI started.")
    print("Face the camera and keep your shoulders visible.")
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
                analysis = {
                    "status": "No body detected",
                    "score": 0,
                    "shoulder_angle": 0,
                    "head_tilt": 0,
                    "feedback": [
                        "Move backward and keep your shoulders visible."
                    ],
                }

            draw_posture_information(frame, analysis)

            cv2.imshow("PostureCheck AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()