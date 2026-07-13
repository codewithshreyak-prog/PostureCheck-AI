import cv2
import textwrap

from src.assessment_engine import PostureAssessmentEngine
from src.feedback import PostureFeedbackGenerator
from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer
from src.posture_smoother import PostureSmoother


def posture_color(score: int):
    """Return an OpenCV color for a posture score."""

    if score >= 85:
        return 0, 255, 0

    if score >= 60:
        return 0, 255, 255

    return 0, 0, 255


def draw_panel_background(
    frame,
    width: int = 710,
    height: int = 430,
) -> None:
    """Draw the transparent information panel."""

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (width, height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )


def draw_line(
    frame,
    text: str,
    y_position: int,
    color=(255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw one line inside the information panel."""

    cv2.putText(
        frame,
        text,
        (25, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_ready_screen(
    frame,
    live_analysis: dict,
) -> None:
    """Display positioning guidance before assessment."""

    draw_panel_background(frame)

    draw_line(
        frame,
        "PostureCheck AI",
        45,
        (255, 255, 255),
        0.82,
        2,
    )

    if live_analysis.get("valid", False):
        draw_line(
            frame,
            "Camera position: Ready",
            85,
            (0, 255, 0),
            0.65,
            2,
        )

        draw_line(
            frame,
            "Your head, shoulders, and hips are visible.",
            122,
        )

        draw_line(
            frame,
            "Press SPACE to begin the posture assessment.",
            160,
            (0, 255, 255),
            0.58,
            2,
        )

        draw_line(
            frame,
            (
                "Shoulder angle: "
                f"{live_analysis['shoulder_angle']:.1f} deg"
            ),
            215,
        )

        draw_line(
            frame,
            (
                "Head tilt: "
                f"{live_analysis['head_tilt']:.1f} deg"
            ),
            245,
        )

        draw_line(
            frame,
            (
                "Hip angle: "
                f"{live_analysis['hip_angle']:.1f} deg"
            ),
            275,
        )

        draw_line(
            frame,
            (
                "Torso lean: "
                f"{live_analysis['torso_lean']:.1f} deg"
            ),
            305,
        )
    else:
        draw_line(
            frame,
            "Camera position: Not ready",
            85,
            (0, 165, 255),
            0.65,
            2,
        )

        feedback = live_analysis.get(
            "feedback",
            ["Move farther back and remain visible."],
        )

        y_position = 130

        for message in feedback[:3]:
            draw_line(
                frame,
                message,
                y_position,
                (0, 165, 255),
            )

            y_position += 35

    draw_line(
        frame,
        "Q: Quit",
        395,
        (255, 255, 255),
        0.5,
    )


def draw_countdown_screen(
    frame,
    assessment_engine,
) -> None:
    """Display the preparation countdown."""

    draw_panel_background(frame)

    countdown = (
        assessment_engine.countdown_remaining()
    )

    draw_line(
        frame,
        "Get Ready",
        75,
        (0, 255, 255),
        0.9,
        2,
    )

    cv2.putText(
        frame,
        str(countdown),
        (300, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        4.5,
        (0, 255, 255),
        8,
    )

    draw_line(
        frame,
        "Stand naturally and look straight ahead.",
        320,
        (255, 255, 255),
        0.62,
        2,
    )

    draw_line(
        frame,
        "Keep your shoulders relaxed.",
        360,
    )


def draw_collection_screen(
    frame,
    assessment_engine,
    live_analysis,
) -> None:
    """Display progress without showing a changing score."""

    draw_panel_background(frame)

    progress = (
        assessment_engine.collection_progress()
    )

    remaining = (
        assessment_engine.collection_remaining()
    )

    draw_line(
        frame,
        "Measuring Posture",
        48,
        (0, 255, 255),
        0.78,
        2,
    )

    draw_line(
        frame,
        f"Time remaining: {remaining:.1f} seconds",
        88,
        (255, 255, 255),
        0.6,
        2,
    )

    bar_left = 25
    bar_top = 120
    bar_width = 620
    bar_height = 28

    cv2.rectangle(
        frame,
        (bar_left, bar_top),
        (
            bar_left + bar_width,
            bar_top + bar_height,
        ),
        (255, 255, 255),
        2,
    )

    filled_width = int(
        bar_width * progress
    )

    cv2.rectangle(
        frame,
        (bar_left, bar_top),
        (
            bar_left + filled_width,
            bar_top + bar_height,
        ),
        (0, 255, 255),
        -1,
    )

    draw_line(
        frame,
        (
            "Valid samples: "
            f"{assessment_engine.valid_sample_count}"
        ),
        190,
    )

    draw_line(
        frame,
        "Hold your normal posture. Avoid deliberate correction.",
        235,
        (255, 255, 255),
        0.55,
    )

    draw_line(
        frame,
        "The final score will appear after measurement.",
        272,
        (255, 255, 255),
        0.55,
    )

    if not live_analysis.get("valid", False):
        draw_line(
            frame,
            "Body visibility reduced. Return to the marked position.",
            325,
            (0, 165, 255),
            0.52,
            2,
        )


def draw_final_screen(
    frame,
    result: dict,
) -> None:
    """Display a compact frozen assessment result."""

    draw_panel_background(
        frame,
        width=570,
        height=485,
    )

    color = posture_color(
        result["score"]
    )

    draw_line(
        frame,
        "Assessment Complete",
        40,
        color,
        0.68,
        2,
    )

    draw_line(
        frame,
        f"Final Score: {result['score']}/100",
        72,
        color,
        0.65,
        2,
    )

    draw_line(
        frame,
        f"Status: {result['status']}",
        102,
        color,
        0.55,
        2,
    )

    draw_line(
        frame,
        (
            f"Confidence: {result['confidence']} "
            f"({result['confidence_score']}%)"
        ),
        132,
        (255, 255, 255),
        0.48,
        1,
    )

    measurements = [
        f"Shoulders: {result['shoulder_angle']:.1f} deg",
        f"Head tilt: {result['head_tilt']:.1f} deg",
        f"Hips: {result['hip_angle']:.1f} deg",
        f"Torso lean: {result['torso_lean']:.1f} deg",
    ]

    y_position = 170

    for measurement in measurements:
        draw_line(
            frame,
            measurement,
            y_position,
            (255, 255, 255),
            0.46,
        )

        y_position += 27

    draw_line(
        frame,
        "Recommendations",
        292,
        color,
        0.52,
        2,
    )

    y_position = 322
    maximum_recommendation_y = 420

    for number, recommendation in enumerate(
        result.get("recommendations", [])[:3],
        start=1,
    ):
        wrapped_lines = textwrap.wrap(
            recommendation,
            width=48,
        )

        for line_index, line in enumerate(
            wrapped_lines
        ):
            if y_position > maximum_recommendation_y:
                break

            if line_index == 0:
                display_text = f"{number}. {line}"
            else:
                display_text = f"   {line}"

            draw_line(
                frame,
                display_text,
                y_position,
                color,
                0.42,
            )

            y_position += 23

        y_position += 3

    draw_line(
        frame,
        "N: New | S: Save | Q: Quit",
        460,
        (0, 255, 255),
        0.43,
        2,
    )


def draw_failed_screen(
    frame,
    result: dict,
) -> None:
    """Display an insufficient-data message."""

    draw_panel_background(frame)

    draw_line(
        frame,
        "Assessment Incomplete",
        60,
        (0, 165, 255),
        0.8,
        2,
    )

    draw_line(
        frame,
        result["message"],
        115,
        (0, 165, 255),
        0.5,
        2,
    )

    draw_line(
        frame,
        (
            f"Valid samples: {result['sample_count']} / "
            f"{result['required_samples']}"
        ),
        165,
    )

    draw_line(
        frame,
        "Press N to reposition and try again.",
        225,
        (0, 255, 255),
        0.58,
        2,
    )


def start_webcam() -> None:
    """Start the reliable timed posture assessment."""

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
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

    assessment_engine = PostureAssessmentEngine(
        countdown_seconds=3.0,
        collection_seconds=8.0,
        sample_interval_seconds=0.1,
        minimum_valid_samples=30,
    )

    feedback_generator = (
        PostureFeedbackGenerator()
    )

    print("PostureCheck AI started.")
    print("SPACE: Start assessment")
    print("N: New assessment")
    print("S: Save completed result")
    print("Q: Quit")

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

            live_analysis = posture_smoother.update(
                raw_analysis
            )

            # Use raw measurements for robust final aggregation.
            assessment_engine.update(
                raw_analysis
            )

            if (
                assessment_engine.state
                == assessment_engine.COMPLETE
                and assessment_engine.final_result
                and "recommendations"
                not in assessment_engine.final_result
            ):
                assessment_engine.final_result[
                    "recommendations"
                ] = feedback_generator.generate(
                    assessment_engine.final_result
                )

            if (
                assessment_engine.state
                == assessment_engine.READY
            ):
                draw_ready_screen(
                    frame,
                    live_analysis,
                )

            elif (
                assessment_engine.state
                == assessment_engine.COUNTDOWN
            ):
                draw_countdown_screen(
                    frame,
                    assessment_engine,
                )

            elif (
                assessment_engine.state
                == assessment_engine.COLLECTING
            ):
                draw_collection_screen(
                    frame,
                    assessment_engine,
                    live_analysis,
                )

            elif (
                assessment_engine.state
                == assessment_engine.COMPLETE
            ):
                draw_final_screen(
                    frame,
                    assessment_engine.final_result,
                )

            elif (
                assessment_engine.state
                == assessment_engine.FAILED
            ):
                draw_failed_screen(
                    frame,
                    assessment_engine.final_result,
                )

            cv2.imshow(
                "PostureCheck AI",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if (
                    assessment_engine.state
                    == assessment_engine.READY
                ):
                    if live_analysis.get(
                        "valid",
                        False,
                    ):
                        assessment_engine.start()

                        print(
                            "Assessment started."
                        )
                    else:
                        print(
                            "Cannot start: keep your head, "
                            "shoulders, and hips visible."
                        )

            if key == ord("n"):
                assessment_engine.reset()
                posture_smoother.reset()

                print(
                    "Ready for a new assessment."
                )

            if key == ord("s"):
                saved_file = (
                    assessment_engine.save_csv()
                )

                if saved_file:
                    print(
                        f"Assessment saved to: {saved_file}"
                    )
                else:
                    print(
                        "Complete an assessment before saving."
                    )

    finally:
        camera.release()
        pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()