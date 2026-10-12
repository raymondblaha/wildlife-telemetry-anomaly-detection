"""Streamlit dashboard for public-safe telemetry anomaly analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from telemetry_anomaly.evaluation import animal_level_flags, row_level_metrics
from telemetry_anomaly.preprocessing import add_movement_features, animal_group_split


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


def contribution_percent(residual_sq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Summarize feature residual contributions as percentages."""

    if residual_sq.size == 0 or not np.any(mask):
        return np.full(residual_sq.shape[1], np.nan, dtype=float)
    mean_residual = np.nanmean(residual_sq[mask], axis=0)
    total = np.nansum(mean_residual)
    if total <= 0:
        return np.full(residual_sq.shape[1], np.nan, dtype=float)
    return mean_residual / total * 100.0


@st.cache_data(show_spinner="Training autoencoder-style model on synthetic reference animals...")
def fit_dashboard_autoencoder(
    df: pd.DataFrame,
    threshold_percentile: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Train a lightweight reconstruction model and score all selected telemetry rows."""

    out = df.copy()
    reference = out[out["label"] == 0].copy()
    if reference["animal_id"].nunique() < 2:
        raise ValueError("Select at least two reference animals so the model can hold one out.")

    reference_train, reference_val = animal_group_split(
        reference,
        group_col="animal_id",
        test_size=0.35,
        random_state=13,
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(reference_train[NUMERIC_FEATURES])
    x_val = scaler.transform(reference_val[NUMERIC_FEATURES])
    x_all = scaler.transform(out[NUMERIC_FEATURES])

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 4, 16, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.01,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        random_state=13,
    )
    model.fit(x_train, x_train)

    recon_all = model.predict(x_all)
    recon_val = model.predict(x_val)
    residual_sq = (x_all - recon_all) ** 2
    val_error = np.sum((x_val - recon_val) ** 2, axis=1)

    threshold = float(np.nanpercentile(val_error, threshold_percentile))
    out["reconstruction_error"] = residual_sq.sum(axis=1)
    out["is_anomaly"] = out["reconstruction_error"] >= threshold

    flagged_mask = out["is_anomaly"].to_numpy(bool)
    comparison_mask = out["label"].eq(1).to_numpy(bool)
    contribution = pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES,
            "flagged_contribution_pct": contribution_percent(residual_sq, flagged_mask),
            "comparison_contribution_pct": contribution_percent(residual_sq, comparison_mask),
            "all_rows_contribution_pct": contribution_percent(
                residual_sq,
                np.ones(len(out), dtype=bool),
            ),
        }
    ).sort_values("flagged_contribution_pct", ascending=False)

    y_true = out["label"].astype(int).to_numpy()
    y_pred = out["is_anomaly"].astype(int).to_numpy()
    metrics = row_level_metrics(y_true, y_pred)
    metrics.update(
        {
            "threshold": threshold,
            "threshold_percentile": int(threshold_percentile),
            "reference_train_animals": int(reference_train["animal_id"].nunique()),
            "reference_validation_animals": int(reference_val["animal_id"].nunique()),
            "reference_train_rows": int(len(reference_train)),
            "reference_validation_rows": int(len(reference_val)),
            "model_iterations": int(model.n_iter_),
            "model_family": "sklearn MLPRegressor reconstruction model",
            "latent_width": 4,
        }
    )
    return out, contribution, metrics


def main() -> None:
    st.set_page_config(
        page_title="Wildlife Telemetry Anomaly Dashboard",
        layout="wide",
    )

    st.title("CWD Movement Anomaly Detection Dashboard")
    st.caption(
        "Public-safe synthetic demo of an autoencoder-style ML workflow for cervid "
        "CWD movement studies."
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
        threshold_percentile = st.slider("Reference validation threshold percentile", 80, 99, 95)
        min_anomalous_fixes = st.slider("Animal flag rule: minimum anomalous fixes", 1, 20, 3)

    filtered = df[
        df["species"].isin(species)
        & df["sex"].isin(sex)
        & df["genotype"].isin(genotype)
    ].copy()
    if filtered.empty:
        st.warning("Select at least one filter value in each sidebar control.")
        st.stop()

    try:
        scored, feature_contribution, model_metrics = fit_dashboard_autoencoder(
            filtered,
            threshold_percentile,
        )
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    st.info(
        "This dashboard trains a lightweight autoencoder-style neural network on synthetic "
        "reference movement data, reconstructs held-out movement features, and flags rows "
        "with high reconstruction error. The production research workflow uses TensorFlow "
        "autoencoders with protected telemetry data."
    )

    animals = scored["animal_id"].nunique()
    fixes = len(scored)
    anomalous_rate = scored["is_anomaly"].mean()
    reference_animals = scored.loc[scored["label"] == 0, "animal_id"].nunique()
    comparison_animals = scored.loc[scored["label"] == 1, "animal_id"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reference Animals", f"{reference_animals:,}")
    c2.metric("Comparison Animals", f"{comparison_animals:,}")
    c3.metric("GPS Fixes", f"{fixes:,}")
    c4.metric("Anomaly Threshold", f"{model_metrics['threshold']:.3f}")
    c5.metric("Flagged Fix Rate", f"{anomalous_rate:.1%}")

    tab_model, tab_animals, tab_contributions, tab_quality, tab_methods = st.tabs(
        [
            "ML Model",
            "Animal Flags",
            "Feature Contributions",
            "Telemetry Quality",
            "Methods",
        ]
    )

    with tab_model:
        st.subheader("Autoencoder-Style Reconstruction Model")
        st.markdown(
            """
            The dashboard now runs the core modeling idea instead of using a placeholder score:

            1. Train only on synthetic reference animals.
            2. Reconstruct movement features from held-out telemetry rows.
            3. Calculate total squared reconstruction error for each GPS fix.
            4. Freeze the anomaly threshold from reference validation animals.
            5. Evaluate row-level and animal-level flags on the selected synthetic cohort.
            """
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Reference Train Animals", f"{model_metrics['reference_train_animals']:,}")
        p2.metric(
            "Reference Validation Animals",
            f"{model_metrics['reference_validation_animals']:,}",
        )
        p3.metric("Latent Width", f"{model_metrics['latent_width']}")
        p4.metric("Training Iterations", f"{model_metrics['model_iterations']:,}")

        fig = px.histogram(
            scored,
            x="reconstruction_error",
            color="label_name",
            nbins=50,
            title="Reconstruction Error Distribution",
            labels={"reconstruction_error": "total squared reconstruction error"},
        )
        fig.add_vline(
            x=model_metrics["threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text="reference validation threshold",
        )
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sensitivity", f"{model_metrics['sensitivity_recall']:.2f}")
        m2.metric("Specificity", f"{model_metrics['specificity']:.2f}")
        m3.metric("Precision", f"{model_metrics['precision']:.2f}")
        m4.metric("Balanced Accuracy", f"{model_metrics['balanced_accuracy']:.2f}")

        cm = confusion_matrix(
            scored["label"].astype(int).to_numpy(),
            scored["is_anomaly"].astype(int).to_numpy(),
            labels=[0, 1],
        )
        cm_df = pd.DataFrame(
            cm,
            index=["Actual reference", "Actual comparison"],
            columns=["Predicted reference", "Predicted anomaly"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            "Synthetic label 0 means reference movement and synthetic label 1 means comparison "
            "movement. These metrics demonstrate the workflow shape, not a biological result."
        )

    with tab_animals:
        st.subheader("Animal-Level Flags")
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
            color="true_label",
            title="Anomalous Fixes per Synthetic Animal",
            labels={"true_label": "synthetic label"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(animal_flags, use_container_width=True)

    with tab_contributions:
        st.subheader("Feature Contributions to Reconstruction Error")
        st.write(
            "This mirrors the publication-style residual summary: calculate squared residuals "
            "for each reconstructed feature, then express each feature's mean residual as a "
            "percentage of total reconstruction error."
        )

        contrib_long = feature_contribution.melt(
            id_vars="feature",
            value_vars=[
                "flagged_contribution_pct",
                "comparison_contribution_pct",
                "all_rows_contribution_pct",
            ],
            var_name="row_subset",
            value_name="contribution_pct",
        )
        subset = st.radio(
            "Contribution subset",
            [
                "flagged_contribution_pct",
                "comparison_contribution_pct",
                "all_rows_contribution_pct",
            ],
            horizontal=True,
        )
        plot_df = (
            contrib_long[contrib_long["row_subset"] == subset]
            .sort_values("contribution_pct", ascending=True)
            .copy()
        )
        fig = px.bar(
            plot_df,
            x="contribution_pct",
            y="feature",
            orientation="h",
            title="Mean Squared Residual Contribution by Feature",
            labels={"contribution_pct": "contribution to reconstruction error (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_contribution, use_container_width=True)
        st.caption(
            "Feature contributions identify which movement dimensions contributed most to "
            "reconstruction error. They are model residual summaries, not causal effects."
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

    with tab_methods:
        st.subheader("Methods and Reproducibility")
        st.markdown(
            "\n".join(
                [
                    "This public dashboard is built around the controls required "
                    "for the research workflow:",
                    "",
                    "- train and validate by animal ID, not only by GPS row;",
                    "- fit preprocessing on reference/training data before applying it elsewhere;",
                    "- reconstruct movement features and score total squared residual error;",
                    "- freeze the anomaly threshold before final evaluation;",
                    "- report row-level and animal-level results separately;",
                    "- inspect feature-level residual contributions for model interpretation;",
                    "- keep raw telemetry and unpublished labels out of GitHub.",
                    "",
                    "The Streamlit dashboard uses a scikit-learn MLP reconstruction model so "
                    "the public demo remains lightweight and easy to run. The protected "
                    "research workflow uses TensorFlow/Keras autoencoders with the study "
                    "telemetry, frozen feature columns, saved scalers, and held-out cohorts.",
                ]
            )
        )

        st.subheader("Public Demo vs. Research Workflow")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "component": "data",
                        "public_dashboard": "synthetic telemetry",
                        "research_workflow": "protected cervid GPS telemetry",
                    },
                    {
                        "component": "model",
                        "public_dashboard": "scikit-learn MLP reconstruction model",
                        "research_workflow": "TensorFlow/Keras autoencoder",
                    },
                    {
                        "component": "threshold",
                        "public_dashboard": "reference validation percentile",
                        "research_workflow": "frozen threshold from training/validation design",
                    },
                    {
                        "component": "interpretation",
                        "public_dashboard": "feature residual contribution percentages",
                        "research_workflow": "holdout residuals and animal-level review",
                    },
                ]
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
