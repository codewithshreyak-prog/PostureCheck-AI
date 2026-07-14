import sqlite3
import textwrap

import cv2

from src.assessment_engine import PostureAssessmentEngine
from src.database import AssessmentDatabase
from src.feedback import PostureFeedbackGenerator
from src.pose_detector import PoseDetector
from src.posture_analyzer import PostureAnalyzer
from src.posture_smoother import PostureSmoother


def posture_color(score: int):
    """Return an OpenCV color based on posture score."""

    if score >= 85:
        return 0, 255, 0

    if score >= 60:
        return 0, 255, 255

    return 0, 0, 255


def draw_panel_background(
    frame,
    width: int = 610,
    height: int = 430,
) -> None:
    """Draw a transparent information panel."""

    frame_height, frame_width = frame.shape[:2]

    panel_right = min(
        width,
        frame_width - 10,
    )

    panel_bottom = min(
        height,
        frame_height - 10,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (panel_right, panel_bottom),
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
    """Draw one line of text inside the panel."""

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
    history_summary: dict,
) -> None:
    """Display camera positioning instructions."""

    draw_panel_background(
        frame,
        width=620,
        height=440,
    )

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
            "Head, shoulders, and hips are visible.",
            120,
        )

        draw_line(
            frame,
            "Press SPACE to begin assessment.",
            158,
            (0, 255, 255),
            0.58,
            2,
        )

        measurements = [
            (
                "Shoulder angle",
                live_analysis["shoulder_angle"],
            ),
            (
                "Head tilt",
                live_analysis["head_tilt"],
            ),
            (
                "Hip angle",
                live_analysis["hip_angle"],
            ),
            (
                "Torso lean",
                live_analysis["torso_lean"],
            ),
        ]

        y_position = 205

        for label, value in measurements:
            draw_line(
                frame,
                f"{label}: {value:.1f} deg",
                y_position,
                (255, 255, 255),
                0.49,
            )

            y_position += 28

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
            [
                "Move farther back and keep your body visible."
            ],
        )

        y_position = 130

        for message in feedback[:3]:
            wrapped_lines = textwrap.wrap(
                message,
                width=55,
            )

            for line in wrapped_lines:
                draw_line(
                    frame,
                    line,
                    y_position,
                    (0, 165, 255),
                    0.5,
                )

                y_position += 29

    draw_line(
        frame,
        (
            "Saved assessments: "
            f"{history_summary['total_assessments']}"
        ),
        365,
        (255, 255, 255),
        0.48,
    )

    draw_line(
        frame,
        (
            "History average: "
            f"{history_summary['average_score']:.1f}"
        ),
        392,
        (255, 255, 255),
        0.48,
    )

    draw_line(
        frame,
        "SPACE: Start | Q: Quit",
        420,
        (0, 255, 255),
        0.46,
        2,
    )


def draw_countdown_screen(
    frame,
    assessment_engine: PostureAssessmentEngine,
) -> None:
    """Display the assessment countdown."""

    draw_panel_background(
        frame,
        width=570,
        height=400,
    )

    countdown = (
        assessment_engine.countdown_remaining()
    )

    draw_line(
        frame,
        "Get Ready",
        70,
        (0, 255, 255),
        0.9,
        2,
    )

    cv2.putText(
        frame,
        str(countdown),
        (250, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        4.5,
        (0, 255, 255),
        8,
    )

    draw_line(
        frame,
        "Stand naturally and look straight ahead.",
        300,
        (255, 255, 255),
        0.58,
        2,
    )

    draw_line(
        frame,
        "Keep your shoulders relaxed.",
        340,
        (255, 255, 255),
        0.55,
    )


def draw_collection_screen(
    frame,
    assessment_engine: PostureAssessmentEngine,
    live_analysis: dict,
) -> None:
    """Display assessment progress without a changing score."""

    draw_panel_background(
        frame,
        width=680,
        height=380,
    )

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
    bar_width = 600
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
        "Hold your natural posture.",
        235,
        (255, 255, 255),
        0.55,
    )

    draw_line(
        frame,
        "The final score appears after measurement.",
        270,
        (255, 255, 255),
        0.55,
    )

    if not live_analysis.get("valid", False):
        draw_line(
            frame,
            "Body visibility reduced. Return to position.",
            325,
            (0, 165, 255),
            0.52,
            2,
        )


def draw_final_screen(
    frame,
    result: dict,
    saved_assessment_id,
) -> None:
    """Display the compact frozen assessment result."""

    draw_panel_background(
        frame,
        width=585,
        height=520,
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
    )

    measurements = [
        (
            "Shoulders",
            result["shoulder_angle"],
        ),
        (
            "Head tilt",
            result["head_tilt"],
        ),
        (
            "Hips",
            result["hip_angle"],
        ),
        (
            "Torso lean",
            result["torso_lean"],
        ),
    ]

    y_position = 170

    for label, value in measurements:
        draw_line(
            frame,
            f"{label}: {value:.1f} deg",
            y_position,
            (255, 255, 255),
            0.46,
        )

        y_position += 27

    draw_line(
        frame,
        "Recommendations",
        290,
        color,
        0.52,
        2,
    )

    y_position = 320

    for number, recommendation in enumerate(
        result.get("recommendations", [])[:3],
        start=1,
    ):
        wrapped_lines = textwrap.wrap(
            recommendation,
            width=49,
        )

        for line_index, line in enumerate(
            wrapped_lines
        ):
            if y_position > 415:
                break

            if line_index == 0:
                display_text = (
                    f"{number}. {line}"
                )
            else:
                display_text = (
                    f"   {line}"
                )

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
        (
            "Accepted frames: "
            f"{result['accepted_percentage']:.1f}%"
        ),
        445,
        (255, 255, 255),
        0.43,
    )

    if saved_assessment_id is None:
        save_status = "Not saved"
        save_color = (0, 165, 255)
    else:
        save_status = (
            f"Saved to database: Record #{saved_assessment_id}"
        )
        save_color = (0, 255, 0)

    draw_line(
        frame,
        save_status,
        472,
        save_color,
        0.42,
        1,
    )

    draw_line(
        frame,
        "N: New | S: Save to DB | Q: Quit",
        500,
        (0, 255, 255),
        0.43,
        2,
    )


def draw_failed_screen(
    frame,
    result: dict,
) -> None:
    """Display an incomplete-assessment message."""

    draw_panel_background(
        frame,
        width=690,
        height=340,
    )

    draw_line(
        frame,
        "Assessment Incomplete",
        60,
        (0, 165, 255),
        0.78,
        2,
    )

    wrapped_message = textwrap.wrap(
        result["message"],
        width=65,
    )

    y_position = 110

    for line in wrapped_message:
        draw_line(
            frame,
            line,
            y_position,
            (0, 165, 255),
            0.48,
            2,
        )

        y_position += 30

    draw_line(
        frame,
        (
            f"Valid samples: {result['sample_count']} / "
            f"{result['required_samples']}"
        ),
        235,
    )

    draw_line(
        frame,
        "Press N to reposition and try again.",
        285,
        (0, 255, 255),
        0.56,
        2,
    )

    draw_line(
        frame,
        "N: New assessment | Q: Quit",
        320,
        (255, 255, 255),
        0.45,
    )


def start_webcam() -> None:
    """Start the PostureCheck AI application."""

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

    database = AssessmentDatabase(
        database_path="data/posturecheck.db"
    )

    saved_assessment_id = None

    history_summary = database.get_summary()

    print("PostureCheck AI started.")
    print("SPACE: Start assessment")
    print("N: New assessment")
    print("S: Save completed result to database")
    print("Q: Quit")
    print(
        "Assessment database: "
        f"{database.database_path}"
    )
    print(
        "Previously saved assessments: "
        f"{history_summary['total_assessments']}"
    )

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
                    history_summary,
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
                    saved_assessment_id,
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
                        saved_assessment_id = None

                        assessment_engine.start()

                        print(
                            "Assessment started."
                        )
                    else:
                        print(
                            "Cannot start assessment. "
                            "Keep your head, shoulders, "
                            "and hips visible."
                        )

            if key == ord("n"):
                assessment_engine.reset()
                posture_smoother.reset()

                saved_assessment_id = None

                history_summary = (
                    database.get_summary()
                )

                print(
                    "Ready for a new assessment."
                )

            if key == ord("s"):
                completed_result = (
                    assessment_engine.final_result
                )

                if (
                    assessment_engine.state
                    != assessment_engine.COMPLETE
                    or not completed_result
                ):
                    print(
                        "Complete an assessment before saving."
                    )

                elif saved_assessment_id is not None:
                    print(
                        "This assessment is already saved "
                        f"as record #{saved_assessment_id}."
                    )

                else:
                    try:
                        saved_assessment_id = (
                            database.save_assessment(
                                completed_result
                            )
                        )

                        completed_result[
                            "database_id"
                        ] = saved_assessment_id

                        history_summary = (
                            database.get_summary()
                        )

                        print(
                            "Assessment saved successfully."
                        )

                        print(
                            "Database record ID: "
                            f"{saved_assessment_id}"
                        )

                        print(
                            "Database location: "
                            f"{database.database_path}"
                        )

                    except (
                        ValueError,
                        sqlite3.Error,
                    ) as error:
                        print(
                            "Unable to save assessment: "
                            f"{error}"
                        )

    finally:
        camera.release()
        pose_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()