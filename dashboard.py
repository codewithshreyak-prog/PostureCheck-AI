from pathlib import Path
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st


DATABASE_PATH = Path("data/posturecheck.db")


st.set_page_config(
    page_title="PostureCheck AI Dashboard",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    rgba(92, 45, 145, 0.18),
                    transparent 30%
                ),
                #0b0b12;
        }

        .dashboard-title {
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0;
        }

        .dashboard-subtitle {
            color: #a7a7b5;
            margin-top: 0.25rem;
            margin-bottom: 1.5rem;
        }

        .section-heading {
            font-size: 1.25rem;
            font-weight: 650;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        .latest-result {
            border: 1px solid rgba(150, 120, 220, 0.35);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            background: rgba(25, 22, 38, 0.8);
        }

        .latest-result strong {
            font-size: 1.05rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(25, 22, 38, 0.75);
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3)
def load_assessments(
    database_path: str,
) -> pd.DataFrame:
    """Load saved posture assessments from SQLite."""

    path = Path(database_path)

    if not path.exists():
        return pd.DataFrame()

    query = """
        SELECT
            id,
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
        FROM assessments
        ORDER BY recorded_at ASC, id ASC
    """

    try:
        with sqlite3.connect(path) as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
            )

    except sqlite3.Error:
        return pd.DataFrame()

    if dataframe.empty:
        return dataframe

    dataframe["recorded_at"] = pd.to_datetime(
        dataframe["recorded_at"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["recorded_at"]
    )

    numeric_columns = [
        "score",
        "confidence_score",
        "shoulder_angle",
        "head_tilt",
        "hip_angle",
        "torso_lean",
        "sample_count",
        "total_sample_slots",
        "accepted_percentage",
        "maximum_dispersion",
        "assessment_duration",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def filter_by_time(
    dataframe: pd.DataFrame,
    selected_period: str,
) -> pd.DataFrame:
    """Filter assessments by the selected time range."""

    if dataframe.empty:
        return dataframe

    if selected_period == "Last 7 days":
        cutoff = pd.Timestamp.now() - pd.Timedelta(
            days=7
        )

        return dataframe[
            dataframe["recorded_at"] >= cutoff
        ]

    if selected_period == "Last 30 days":
        cutoff = pd.Timestamp.now() - pd.Timedelta(
            days=30
        )

        return dataframe[
            dataframe["recorded_at"] >= cutoff
        ]

    if selected_period == "Last 90 days":
        cutoff = pd.Timestamp.now() - pd.Timedelta(
            days=90
        )

        return dataframe[
            dataframe["recorded_at"] >= cutoff
        ]

    return dataframe


def calculate_score_change(
    dataframe: pd.DataFrame,
):
    """Calculate change between the two latest scores."""

    if len(dataframe) < 2:
        return None

    latest_score = float(
        dataframe.iloc[-1]["score"]
    )

    previous_score = float(
        dataframe.iloc[-2]["score"]
    )

    return round(
        latest_score - previous_score,
        1,
    )


def render_metric_cards(
    dataframe: pd.DataFrame,
) -> None:
    """Display the main assessment statistics."""

    scores = dataframe["score"].dropna()

    total_assessments = len(dataframe)
    average_score = scores.mean()
    best_score = scores.max()
    latest_score = scores.iloc[-1]

    score_change = calculate_score_change(
        dataframe
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        label="Total Assessments",
        value=total_assessments,
        border=True,
    )

    metric_columns[1].metric(
        label="Latest Score",
        value=f"{latest_score:.0f}/100",
        delta=score_change,
        delta_description="vs previous",
        border=True,
        chart_data=scores.tolist(),
        chart_type="line",
    )

    metric_columns[2].metric(
        label="Average Score",
        value=f"{average_score:.1f}/100",
        border=True,
        chart_data=scores.tolist(),
        chart_type="area",
    )

    metric_columns[3].metric(
        label="Best Score",
        value=f"{best_score:.0f}/100",
        border=True,
    )


def render_latest_assessment(
    dataframe: pd.DataFrame,
) -> None:
    """Display the most recently saved assessment."""

    latest = dataframe.iloc[-1]

    recorded_time = latest["recorded_at"].strftime(
        "%b %d, %Y — %I:%M %p"
    )

    recommendations = latest.get(
        "recommendations",
        "",
    )

    if not recommendations:
        recommendations = (
            "No recommendations were stored."
        )

    st.markdown(
        '<div class="section-heading">'
        "Latest Assessment"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="latest-result">
            <strong>
                {latest["status"]} — {latest["score"]:.0f}/100
            </strong>
            <br><br>
            Recorded: {recorded_time}
            <br>
            Confidence:
            {latest["confidence"]}
            ({latest["confidence_score"]:.0f}%)
            <br>
            Accepted frames:
            {latest["accepted_percentage"]:.1f}%
            <br><br>
            <strong>Recommendation</strong>
            <br>
            {recommendations}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_trend(
    dataframe: pd.DataFrame,
) -> None:
    """Display posture scores over time."""

    score_figure = px.line(
        dataframe,
        x="recorded_at",
        y="score",
        markers=True,
        labels={
            "recorded_at": "Assessment time",
            "score": "Posture score",
        },
        title="Posture Score Trend",
    )

    score_figure.update_yaxes(
        range=[0, 100],
        dtick=10,
    )

    score_figure.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        hovermode="x unified",
    )

    score_figure.add_hline(
        y=85,
        line_dash="dash",
        annotation_text="Good posture",
    )

    score_figure.add_hline(
        y=60,
        line_dash="dash",
        annotation_text="Moderate posture",
    )

    st.plotly_chart(
        score_figure,
        width="stretch",
        height=390,
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )


def render_alignment_trends(
    dataframe: pd.DataFrame,
) -> None:
    """Display body-alignment measurements over time."""

    measurement_data = dataframe[
        [
            "recorded_at",
            "shoulder_angle",
            "head_tilt",
            "hip_angle",
            "torso_lean",
        ]
    ].copy()

    measurement_data = measurement_data.rename(
        columns={
            "shoulder_angle": "Shoulder angle",
            "head_tilt": "Head tilt",
            "hip_angle": "Hip angle",
            "torso_lean": "Torso lean",
        }
    )

    long_data = measurement_data.melt(
        id_vars="recorded_at",
        var_name="Measurement",
        value_name="Angle",
    )

    alignment_figure = px.line(
        long_data,
        x="recorded_at",
        y="Angle",
        color="Measurement",
        markers=True,
        title="Body Alignment Trends",
        labels={
            "recorded_at": "Assessment time",
            "Angle": "Angle in degrees",
        },
    )

    alignment_figure.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        hovermode="x unified",
    )

    alignment_figure.add_hline(
        y=0,
        line_dash="dash",
    )

    st.plotly_chart(
        alignment_figure,
        width="stretch",
        height=390,
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )


def render_status_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Display assessment-status distribution."""

    status_counts = (
        dataframe["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Assessments")
    )

    status_figure = px.bar(
        status_counts,
        x="Status",
        y="Assessments",
        text="Assessments",
        title="Posture Status Distribution",
    )

    status_figure.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    status_figure.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        status_figure,
        width="stretch",
        height=350,
        config={
            "displaylogo": False,
        },
    )


def render_confidence_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Display assessment confidence distribution."""

    confidence_counts = (
        dataframe["confidence"]
        .value_counts()
        .rename_axis("Confidence")
        .reset_index(name="Assessments")
    )

    confidence_figure = px.pie(
        confidence_counts,
        names="Confidence",
        values="Assessments",
        hole=0.55,
        title="Confidence Distribution",
    )

    confidence_figure.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        confidence_figure,
        width="stretch",
        height=350,
        config={
            "displaylogo": False,
        },
    )


def render_recent_assessments(
    dataframe: pd.DataFrame,
) -> None:
    """Display recent assessment records."""

    recent = dataframe.sort_values(
        by="recorded_at",
        ascending=False,
    ).head(20).copy()

    recent["recorded_at"] = recent[
        "recorded_at"
    ].dt.strftime(
        "%b %d, %Y %I:%M %p"
    )

    recent = recent.rename(
        columns={
            "recorded_at": "Recorded At",
            "score": "Score",
            "status": "Status",
            "confidence": "Confidence",
            "confidence_score": "Confidence %",
            "shoulder_angle": "Shoulders",
            "head_tilt": "Head Tilt",
            "hip_angle": "Hips",
            "torso_lean": "Torso Lean",
            "accepted_percentage": "Accepted Frames %",
            "recommendations": "Recommendations",
        }
    )

    displayed_columns = [
        "Recorded At",
        "Score",
        "Status",
        "Confidence",
        "Confidence %",
        "Shoulders",
        "Head Tilt",
        "Hips",
        "Torso Lean",
        "Accepted Frames %",
        "Recommendations",
    ]

    st.dataframe(
        recent[displayed_columns],
        width="stretch",
        height=430,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Confidence %":
                st.column_config.ProgressColumn(
                    "Confidence %",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            "Shoulders":
                st.column_config.NumberColumn(
                    "Shoulders",
                    format="%.1f°",
                ),
            "Head Tilt":
                st.column_config.NumberColumn(
                    "Head Tilt",
                    format="%.1f°",
                ),
            "Hips":
                st.column_config.NumberColumn(
                    "Hips",
                    format="%.1f°",
                ),
            "Torso Lean":
                st.column_config.NumberColumn(
                    "Torso Lean",
                    format="%.1f°",
                ),
            "Accepted Frames %":
                st.column_config.NumberColumn(
                    "Accepted Frames %",
                    format="%.1f%%",
                ),
        },
    )


def main() -> None:
    """Render the posture-assessment dashboard."""

    st.markdown(
        '<div class="dashboard-title">'
        "PostureCheck AI Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        "Track posture assessments, alignment trends, "
        "confidence, and progress over time."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Dashboard Controls")

        if st.button(
            "Refresh database",
            width="stretch",
        ):
            st.cache_data.clear()
            st.rerun()

        selected_period = st.selectbox(
            "Time period",
            options=[
                "All time",
                "Last 7 days",
                "Last 30 days",
                "Last 90 days",
            ],
        )

        st.divider()

        st.caption(
            f"Database: {DATABASE_PATH}"
        )

    dataframe = load_assessments(
        str(DATABASE_PATH)
    )

    if dataframe.empty:
        st.warning(
            "No saved assessments were found."
        )

        st.code(
            "python app.py\n"
            "# Complete an assessment and press S\n"
            "streamlit run dashboard.py",
            language="bash",
        )

        return

    dataframe = filter_by_time(
        dataframe,
        selected_period,
    )

    available_statuses = sorted(
        dataframe["status"]
        .dropna()
        .unique()
        .tolist()
    )

    with st.sidebar:
        selected_statuses = st.multiselect(
            "Posture status",
            options=available_statuses,
            default=available_statuses,
        )

    if selected_statuses:
        dataframe = dataframe[
            dataframe["status"].isin(
                selected_statuses
            )
        ]

    if dataframe.empty:
        st.info(
            "No assessments match the current filters."
        )
        return

    render_metric_cards(
        dataframe
    )

    st.divider()

    render_latest_assessment(
        dataframe
    )

    st.divider()

    trend_column, alignment_column = st.columns(
        2
    )

    with trend_column:
        render_score_trend(
            dataframe
        )

    with alignment_column:
        render_alignment_trends(
            dataframe
        )

    distribution_column, confidence_column = (
        st.columns(2)
    )

    with distribution_column:
        render_status_distribution(
            dataframe
        )

    with confidence_column:
        render_confidence_distribution(
            dataframe
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">'
        "Recent Assessments"
        "</div>",
        unsafe_allow_html=True,
    )

    render_recent_assessments(
        dataframe
    )

    export_data = dataframe.copy()

    export_data["recorded_at"] = export_data[
        "recorded_at"
    ].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    csv_data = export_data.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="Download filtered history as CSV",
        data=csv_data,
        file_name="posturecheck_history.csv",
        mime="text/csv",
        width="stretch",
    )

    st.caption(
        "PostureCheck AI is an educational posture-analysis "
        "prototype and is not a medical diagnostic system."
    )


if __name__ == "__main__":
    main()