"""Streamlit dashboard for public-safe telemetry anomaly analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from telemetry_anomaly.evaluation import animal_level_flags, row_level_metrics
from telemetry_anomaly.preprocessing import add_movement_features


SAMPLE_PATH = ROOT / "data" / "sample" / "synthetic_telemetry.csv"
NUMERIC_FEATURES = [
    "step_distance_m",
    "elapsed_hours",
    "speed_m_per_hour",
    "turning_angle_deg",
    "rolling_3_step_mean_m",
    "rolling_3_speed_mean",
    "hour",
    "month",
]


@st.cache_data
def load_dashboard_data(path: str) -> pd.DataFrame:
    """Load telemetry and add public-safe movement features."""

    csv_path = Path(path)
    if not csv_path.exists():
        st.error(
            "Sample data not found. Run `python scripts/generate_sample_data.py` "
            "from the repository root."
        )
        st.stop()

    raw = pd.read_csv(csv_path)
    featured = add_movement_features(raw).dropna(subset=NUMERIC_FEATURES + ["label"]).copy()
    featured["label_name"] = np.where(featured["label"] == 1, "comparison", "reference")
    return featured


def add_demo_anomaly_scores(df: pd.DataFrame, threshold_percentile: int) -> tuple[pd.DataFrame, float]:
    """Create a transparent demo score so the dashboard runs without TensorFlow."""

    out = df.copy()
    reference = out[out["label"] == 0]
    scaler = StandardScaler()
    scaler.fit(reference[NUMERIC_FEATURES])

    scaled = scaler.transform(out[NUMERIC_FEATURES])
    center = scaler.transform(reference[NUMERIC_FEATURES]).mean(axis=0)
    out["demo_anomaly_score"] = np.mean((scaled - center) ** 2, axis=1)

    threshold_source = out[out["label"] == 0]["demo_anomaly_score"]
    threshold = float(np.percentile(threshold_source, threshold_percentile))
    out["is_anomaly"] = out["demo_anomaly_score"] > threshold
    return out, threshold


def main() -> None:
    st.set_page_config(
        page_title="Wildlife Telemetry Anomaly Dashboard",
        layout="wide",
    )

    st.title("Wildlife Telemetry Anomaly Dashboard")
    st.caption(
        "Public-safe dashboard using synthetic telemetry to demonstrate data-quality, "
        "movement-feature, and anomaly-evaluation workflows."
    )

    df = load_dashboard_data(str(SAMPLE_PATH))

    with st.sidebar:
        st.header("Filters")
        species = st.multiselect(
            "Species",
            sorted(df["species"].unique()),
            default=sorted(df["species"].unique()),
        )
        sex = st.multiselect("Sex", sorted(df["sex"].unique()), default=sorted(df["sex"].unique()))
        genotype = st.multiselect(
            "Genotype",
            sorted(df["genotype"].unique()),
            default=sorted(df["genotype"].unique()),
        )
        threshold_percentile = st.slider("Reference threshold percentile", 80, 99, 95)
        min_anomalous_fixes = st.slider("Animal flag rule: minimum anomalous fixes", 1, 20, 3)

    filtered = df[
        df["species"].isin(species)
        & df["sex"].isin(sex)
        & df["genotype"].isin(genotype)
    ].copy()
    scored, threshold = add_demo_anomaly_scores(filtered, threshold_percentile)

    st.info(
        "This dashboard uses synthetic data and a transparent demo anomaly score. "
        "The production workflow in this repo uses a frozen-threshold autoencoder."
    )

    animals = scored["animal_id"].nunique()
    fixes = len(scored)
    anomalous_rate = scored["is_anomaly"].mean()
    reference_animals = scored.loc[scored["label"] == 0, "animal_id"].nunique()
    comparison_animals = scored.loc[scored["label"] == 1, "animal_id"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Animals", f"{animals:,}")
    c2.metric("GPS Fixes", f"{fixes:,}")
    c3.metric("Reference Animals", f"{reference_animals:,}")
    c4.metric("Comparison Animals", f"{comparison_animals:,}")
    c5.metric("Anomalous Fix Rate", f"{anomalous_rate:.1%}")

    tab_quality, tab_features, tab_model, tab_animals, tab_methods = st.tabs(
        [
            "Telemetry Quality",
            "Feature Explorer",
            "Threshold Review",
            "Animal Review",
            "Methods",
        ]
    )

    with tab_quality:
        st.subheader("Telemetry Data Quality")
        scored["date"] = pd.to_datetime(scored["timestamp"]).dt.date
        daily_counts = (
            scored.groupby(["animal_id", "date", "species", "label_name"])
            .size()
            .reset_index(name="fixes_per_day")
        )

        q1, q2 = st.columns([1, 1])
        with q1:
            fig = px.histogram(
                daily_counts,
                x="fixes_per_day",
                color="species",
                barmode="group",
                title="Fixes per Animal-Day",
            )
            st.plotly_chart(fig, use_container_width=True)
        with q2:
            retention = (
                daily_counts.assign(retained=lambda x: x["fixes_per_day"] >= 2)
                .groupby(["species", "label_name"])["retained"]
                .mean()
                .reset_index()
            )
            fig = px.bar(
                retention,
                x="species",
                y="retained",
                color="label_name",
                text_auto=".1%",
                title="Demo Retention Rate: At Least 2 Fixes per Day",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            daily_counts.groupby(["species", "label_name"])["fixes_per_day"]
            .describe()
            .reset_index(),
            use_container_width=True,
        )

    with tab_features:
        st.subheader("Movement Feature Explorer")
        feature = st.selectbox(
            "Feature",
            [
                "speed_m_per_hour",
                "step_distance_m",
                "turning_angle_deg",
                "rolling_3_speed_mean",
                "rolling_3_step_mean_m",
            ],
        )

        f1, f2 = st.columns([1, 1])
        with f1:
            fig = px.histogram(
                scored,
                x=feature,
                color="label_name",
                marginal="box",
                nbins=45,
                title=f"{feature} Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        with f2:
            fig = px.box(
                scored,
                x="species",
                y=feature,
                color="label_name",
                points=False,
                title=f"{feature} by Species and Label",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            scored.groupby(["species", "sex", "genotype", "label_name"])[feature]
            .describe()
            .reset_index(),
            use_container_width=True,
        )

    with tab_model:
        st.subheader("Threshold Review")
        st.write(
            "The dashboard demonstrates the same evaluation habit used in the modeling workflow: "
            "choose a threshold on reference data, then evaluate held-out comparison data."
        )

        fig = px.histogram(
            scored,
            x="demo_anomaly_score",
            color="label_name",
            nbins=50,
            title="Demo Anomaly Score Distribution",
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        y_true = scored["label"].astype(int).to_numpy()
        y_pred = scored["is_anomaly"].astype(int).to_numpy()
        metrics = row_level_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sensitivity", f"{metrics['sensitivity_recall']:.2f}")
        m2.metric("Specificity", f"{metrics['specificity']:.2f}")
        m3.metric("Precision", f"{metrics['precision']:.2f}")
        m4.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.2f}")

        cm_df = pd.DataFrame(
            cm,
            index=["Actual reference", "Actual comparison"],
            columns=["Predicted reference", "Predicted anomaly"],
        )
        st.dataframe(cm_df, use_container_width=True)

    with tab_animals:
        st.subheader("Animal-Level Review")
        animal_flags = animal_level_flags(
            scored,
            animal_col="animal_id",
            label_col="label",
            flag_col="is_anomaly",
            min_anomalous_fixes=min_anomalous_fixes,
        )
        animal_flags = animal_flags.merge(
            scored[["animal_id", "species", "sex", "genotype"]].drop_duplicates(),
            on="animal_id",
            how="left",
        )
        animal_flags["anomalous_fix_rate"] = (
            animal_flags["n_anomalous_fixes"] / animal_flags["n_fixes"]
        )

        fig = px.bar(
            animal_flags.sort_values("n_anomalous_fixes", ascending=False),
            x="animal_id",
            y="n_anomalous_fixes",
            color="species",
            title="Anomalous Fixes per Synthetic Animal",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(animal_flags, use_container_width=True)

    with tab_methods:
        st.subheader("Methods and Reproducibility")
        st.markdown(
            """
            This public dashboard is built around the same controls required for the research workflow:

            - split and evaluate by animal ID, not only by GPS row;
            - fit preprocessing on reference/training data before applying it elsewhere;
            - freeze the threshold before final evaluation;
            - report row-level and animal-level results separately;
            - keep raw telemetry, exact coordinates, and unpublished animal labels out of GitHub.

            The dashboard is intentionally synthetic. It demonstrates the analysis interface and
            quality-control workflow without exposing protected research data.
            """
        )


if __name__ == "__main__":
    main()
