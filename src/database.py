import sqlite3
from datetime import datetime
from pathlib import Path


class AssessmentDatabase:
    """Store and retrieve finalized posture assessments using SQLite."""

    def __init__(
        self,
        database_path: str = "data/posturecheck.db",
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        """Create a configured SQLite database connection."""

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self) -> None:
        """Create the assessments table if it does not exist."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    confidence_score INTEGER NOT NULL,
                    shoulder_angle REAL NOT NULL,
                    head_tilt REAL NOT NULL,
                    hip_angle REAL NOT NULL,
                    torso_lean REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    total_sample_slots INTEGER NOT NULL,
                    accepted_percentage REAL NOT NULL,
                    maximum_dispersion REAL NOT NULL,
                    assessment_duration REAL NOT NULL,
                    recommendations TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_assessments_recorded_at
                ON assessments(recorded_at)
                """
            )

    def save_assessment(
        self,
        result: dict,
    ) -> int:
        """Save one completed posture assessment."""

        if not result or not result.get("valid", False):
            raise ValueError(
                "Only valid completed assessments can be saved."
            )

        recommendations = " | ".join(
            result.get("recommendations", [])
        )

        recorded_at = datetime.now().isoformat(
            timespec="seconds"
        )

        values = (
            recorded_at,
            int(result["score"]),
            str(result["status"]),
            str(result["confidence"]),
            int(result["confidence_score"]),
            float(result["shoulder_angle"]),
            float(result["head_tilt"]),
            float(result["hip_angle"]),
            float(result["torso_lean"]),
            int(result["sample_count"]),
            int(result["total_sample_slots"]),
            float(result["accepted_percentage"]),
            float(result["maximum_dispersion"]),
            float(result["assessment_duration"]),
            recommendations,
        )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO assessments (
                    recorded_at,
                    score,
                    status,
                    confidence,
                    confidence_score,
                    shoulder_angle,
                    head_tilt,
                    hip_angle,
                    torso_lean,
                    sample_count,
                    total_sample_slots,
                    accepted_percentage,
                    maximum_dispersion,
                    assessment_duration,
                    recommendations
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

            assessment_id = cursor.lastrowid

        return int(assessment_id)

    def get_recent_assessments(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """Return the most recently saved assessments."""

        safe_limit = max(
            1,
            min(int(limit), 100),
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM assessments
                ORDER BY recorded_at DESC, id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def get_summary(self) -> dict:
        """Return overall assessment-history statistics."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_assessments,
                    AVG(score) AS average_score,
                    MAX(score) AS best_score,
                    MIN(score) AS lowest_score,
                    SUM(
                        CASE
                            WHEN score >= 85 THEN 1
                            ELSE 0
                        END
                    ) AS good_assessments,
                    SUM(
                        CASE
                            WHEN score >= 60
                            AND score < 85 THEN 1
                            ELSE 0
                        END
                    ) AS moderate_assessments,
                    SUM(
                        CASE
                            WHEN score < 60 THEN 1
                            ELSE 0
                        END
                    ) AS poor_assessments
                FROM assessments
                """
            ).fetchone()

        total_assessments = int(
            row["total_assessments"] or 0
        )

        return {
            "total_assessments": total_assessments,
            "average_score": round(
                float(row["average_score"] or 0.0),
                1,
            ),
            "best_score": int(
                row["best_score"] or 0
            ),
            "lowest_score": int(
                row["lowest_score"] or 0
            ),
            "good_assessments": int(
                row["good_assessments"] or 0
            ),
            "moderate_assessments": int(
                row["moderate_assessments"] or 0
            ),
            "poor_assessments": int(
                row["poor_assessments"] or 0
            ),
        }