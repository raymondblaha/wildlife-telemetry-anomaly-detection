"""Streamlit dashboard for manuscript-profile telemetry anomaly outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
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
from telemetry_anomaly.manuscript_profiles import get_profile
from telemetry_anomaly.preprocessing import add_movement_features, animal_group_split


SAMPLE_PATH = ROOT / "data" / "sample" / "synthetic_telemetry.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs" / "wtd_primary"
SYNTHETIC_FEATURES = [
    "step_distance_m",
    "elapsed_hours",
    "speed_m_per_hour",
    "turning_angle_deg",
    "rolling_3_step_mean_m",
    "rolling_3_speed_mean",
    "hour",
    "month",
]
FEATURE_GLOSSARY = {
    "velo": ("Velocity", "Movement speed between fixes."),
    "abs_rel_angle": ("Turning", "Absolute relative turn angle."),
    "sinuosity": ("Path shape", "Path tortuosity for a movement segment."),
    "sinuosity_28pt": ("Path shape", "Mule deer 28-point path tortuosity metric."),
    "rolling_abs_rel_angle_mean": ("Rolling turning", "Rolling mean turn angle."),
    "rolling_abs_rel_angle_sd": ("Rolling turning", "Rolling turn-angle variability."),
    "rolling_abs_rel_angle_skew": ("Rolling turning", "Rolling turn-angle skewness."),
    "rolling_abs_rel_angle_kurt": ("Rolling turning", "Rolling turn-angle kurtosis."),
    "rolling_velo_mean": ("Rolling velocity", "Rolling mean velocity."),
    "rolling_velo_sd": ("Rolling velocity", "Rolling velocity variability."),
    "rolling_velo_skew": ("Rolling velocity", "Rolling velocity skewness."),
    "rolling_velo_kurt": ("Rolling velocity", "Rolling velocity kurtosis."),
    "akde_low": ("Home range", "Lower AKDE estimate."),
    "akde_est": ("Home range", "Central AKDE estimate."),
    "akde_high": ("Home range", "Upper AKDE estimate."),
    "low": ("Home range", "Lower home-range estimate from mule deer workflow."),
    "est": ("Home range", "Central home-range estimate from mule deer workflow."),
    "high": ("Home range", "Upper home-range estimate from mule deer workflow."),
    "season": ("Temporal", "Season category encoded from telemetry date."),
}


def required_artifacts(profile_name: str, artifact_dir: Path) -> dict[str, Path]:
    profile = get_profile(profile_name)
    suffix = profile.processed_suffix
    return {
        "feature_columns": artifact_dir / f"feature_columns_{suffix}.pkl",
        "train": artifact_dir / f"train_processed_{suffix}.csv",
        "validation": artifact_dir / f"val_processed_{suffix}.csv",
        "test": artifact_dir / f"test_processed_{suffix}.csv",
        "threshold": artifact_dir / "reconstruction_threshold.pkl",
        "id_rule": artifact_dir / "id_rule_params.pkl",
        "rows": artifact_dir / f"test_processed_{suffix}_rows_with_flags_FROZEN.csv",
        "residuals": artifact_dir / f"test_processed_{suffix}_rows_with_recon_and_residuals.csv",
        "id_summary": artifact_dir / f"test_processed_{suffix}_id_summary_FROZEN.csv",
        "contrib": artifact_dir / "ae_feature_contributions_holdout.csv",
        "contrib_summary": artifact_dir / "ae_feature_contribution_summary_holdout.csv",
        "history": artifact_dir / "training_history.npy",
        "model": artifact_dir / "autoencoder_model.h5",
    }


def existing_artifacts(paths: dict[str, Path]) -> dict[str, bool]:
    return {name: path.exists() for name, path in paths.items()}


@st.cache_data
def load_optional_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_data
def load_optional_joblib(path: str) -> Any | None:
    pkl_path = Path(path)
    if not pkl_path.exists():
        return None
    return joblib.load(pkl_path)


@st.cache_data
def load_training_history(path: str) -> pd.DataFrame | None:
    history_path = Path(path)
    if not history_path.exists():
        return None
    raw = np.load(history_path, allow_pickle=True).item()
    history = pd.DataFrame(raw)
    history["epoch"] = np.arange(1, len(history) + 1)
    return history


@st.cache_resource(show_spinner=False)
def load_weight_summary(model_path: str) -> pd.DataFrame | None:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None

    model = load_model(path, compile=False)
    rows = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        rows.append(
            {
                "layer": layer.name,
                "shape": "x".join(str(x) for x in kernel.shape),
                "mean_abs_weight": float(np.mean(np.abs(kernel))),
                "max_abs_weight": float(np.max(np.abs(kernel))),
                "nonzero_weight_rate": float(np.mean(np.abs(kernel) > 1e-12)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_synthetic_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        st.error(
            "Synthetic data not found. Run `python scripts/generate_sample_data.py` "
            "from the repository root."
        )
        st.stop()

    raw = pd.read_csv(csv_path)
    featured = add_movement_features(raw).dropna(subset=SYNTHETIC_FEATURES + ["label"])
    featured = featured.copy()
    featured["label_name"] = np.where(featured["label"] == 1, "comparison", "reference")
    return featured


def contribution_percent(residual_sq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if residual_sq.size == 0 or not np.any(mask):
        return np.full(residual_sq.shape[1], np.nan, dtype=float)
    means = np.nanmean(residual_sq[mask], axis=0)
    total = np.nansum(means)
    return (means / total) * 100.0 if total > 0 else np.full(residual_sq.shape[1], np.nan)


@st.cache_data(show_spinner="Training reduced synthetic reconstruction model...")
def fit_synthetic_demo(
    df: pd.DataFrame,
    threshold_percentile: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    out = df.copy()
    reference = out[out["label"] == 0].copy()
    if reference["animal_id"].nunique() < 2:
        raise ValueError("Select at least two reference animals.")

    reference_train, reference_val = animal_group_split(
        reference,
        group_col="animal_id",
        test_size=0.35,
        random_state=13,
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(reference_train[SYNTHETIC_FEATURES])
    x_val = scaler.transform(reference_val[SYNTHETIC_FEATURES])
    x_all = scaler.transform(out[SYNTHETIC_FEATURES])

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
    out["row_flag"] = out["reconstruction_error"] >= threshold
    flagged_mask = out["row_flag"].to_numpy(bool)
    comparison_mask = out["label"].eq(1).to_numpy(bool)
    contribution = pd.DataFrame(
        {
            "feature": SYNTHETIC_FEATURES,
            "Flagged_%": contribution_percent(residual_sq, flagged_mask),
            "Comparison_%": contribution_percent(residual_sq, comparison_mask),
            "All_%": contribution_percent(residual_sq, np.ones(len(out), dtype=bool)),
        }
    ).sort_values("Flagged_%", ascending=False)

    y_true = out["label"].astype(int).to_numpy()
    y_pred = out["row_flag"].astype(int).to_numpy()
    metrics = row_level_metrics(y_true, y_pred)
    sweep = []
    for percentile in range(80, 100):
        cutoff = float(np.nanpercentile(val_error, percentile))
        pred = out["reconstruction_error"].ge(cutoff).astype(int).to_numpy()
        sweep_metrics = row_level_metrics(y_true, pred)
        sweep.append(
            {
                "percentile": percentile,
                "threshold": cutoff,
                "flagged_fix_rate": float(pred.mean()),
                "balanced_accuracy": sweep_metrics["balanced_accuracy"],
                "sensitivity": sweep_metrics["sensitivity_recall"],
                "specificity": sweep_metrics["specificity"],
            }
        )
    weight_summary = pd.DataFrame(
        [
            {
                "layer": f"dense_{i + 1}",
                "shape": "x".join(str(x) for x in coef.shape),
                "mean_abs_weight": float(np.mean(np.abs(coef))),
                "max_abs_weight": float(np.max(np.abs(coef))),
                "nonzero_weight_rate": float(np.mean(np.abs(coef) > 1e-12)),
            }
            for i, coef in enumerate(model.coefs_)
        ]
    )
    metrics.update(
        {
            "threshold": threshold,
            "threshold_percentile": int(threshold_percentile),
            "reference_train_animals": int(reference_train["animal_id"].nunique()),
            "reference_validation_animals": int(reference_val["animal_id"].nunique()),
            "reference_train_rows": int(len(reference_train)),
            "reference_validation_rows": int(len(reference_val)),
            "model_iterations": int(model.n_iter_),
            "model_family": "synthetic sklearn reconstruction model",
            "latent_width": 4,
        }
    )
    return out, contribution, pd.DataFrame(sweep), weight_summary, metrics


def feature_table(profile_name: str, feature_columns: list[str] | None) -> pd.DataFrame:
    profile = get_profile(profile_name)
    rows = []
    base_features = profile.numeric_features + profile.categorical_features
    for feature in base_features:
        family, description = FEATURE_GLOSSARY.get(feature, ("Encoded", "Model input feature."))
        rows.append(
            {
                "feature": feature,
                "family": family,
                "description": description,
                "from_profile": True,
                "present_in_artifact": feature_columns is None or feature in feature_columns,
            }
        )
    if feature_columns:
        for feature in feature_columns:
            if feature not in base_features and feature.startswith("season_"):
                rows.append(
                    {
                        "feature": feature,
                        "family": "Temporal",
                        "description": "One-hot encoded season column.",
                        "from_profile": False,
                        "present_in_artifact": True,
                    }
                )
    return pd.DataFrame(rows)


def artifact_status_table(paths: dict[str, Path], exists: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "status": "found" if exists[name] else "missing",
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            }
            for name, path in paths.items()
        ]
    )


def show_feature_tab(
    profile_name: str,
    feature_columns: list[str] | None,
    processed: pd.DataFrame | None,
) -> None:
    st.subheader("1. Preprocessing Results")
    st.write(
        "This tab shows the feature matrix produced before model training. The synthetic "
        "pipeline follows the manuscript flow: generate public-safe telemetry-shaped inputs, "
        "clean rows, fit the scaler/season encoder on reference training animals only, then "
        "write train, validation, and test feature files for the AE."
    )
    st.caption(
        "Key movement metrics include velocity, turning angle, sinuosity, rolling velocity "
        "moments, rolling turn-angle moments, AKDE/home-range estimates, and season encodings."
    )
    features = feature_table(profile_name, feature_columns)
    st.dataframe(features, use_container_width=True)

    if processed is None:
        st.info("Processed training/test CSVs were not found, so distributions are hidden.")
        return

    numeric_cols = [c for c in features["feature"].tolist() if c in processed.columns]
    if not numeric_cols:
        return
    selected = st.selectbox("Inspect feature distribution", numeric_cols)
    fig = px.histogram(
        processed,
        x=selected,
        nbins=50,
        title=f"{selected} Distribution in Loaded Processed Rows",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(processed[numeric_cols].describe().T.reset_index(), use_container_width=True)


def show_model_tab(
    profile_name: str,
    threshold_info: dict[str, Any] | None,
    id_rule: dict[str, Any] | None,
    history: pd.DataFrame | None,
    weight_summary: pd.DataFrame | None,
    synthetic_metrics: dict[str, Any] | None = None,
    synthetic_weights: pd.DataFrame | None = None,
) -> None:
    st.subheader("2. Training the Autoencoder")
    st.write(
        "The autoencoder is trained on reference movement. It compresses each row into a "
        "small bottleneck, reconstructs the movement features, and treats high reconstruction "
        "error as evidence that a row does not look like the learned reference pattern."
    )

    profile = get_profile(profile_name)
    architecture = pd.DataFrame(
        [
            {"stage": "input", "width": "feature count", "purpose": "scaled movement features"},
            {"stage": "encoder_1", "width": 32, "purpose": "learn broad movement patterns"},
            {"stage": "encoder_2", "width": 16, "purpose": "compress stable patterns"},
            {"stage": "bottleneck", "width": 4, "purpose": "low-dimensional reference signal"},
            {"stage": "decoder_1", "width": 16, "purpose": "reconstruct compressed signal"},
            {"stage": "decoder_2", "width": 32, "purpose": "reconstruct feature detail"},
            {"stage": "output", "width": "feature count", "purpose": "predicted feature values"},
        ]
    )
    st.dataframe(architecture, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profile", profile.name)
    c2.metric("Default LR", f"{profile.default_learning_rate:g}")
    c3.metric("Default Batch", f"{profile.default_batch_size}")
    c4.metric("Default Count Rule", f"{profile.default_count_rule}")

    if threshold_info:
        st.subheader("Frozen Threshold")
        st.dataframe(pd.DataFrame([threshold_info]), use_container_width=True)
    if id_rule:
        st.subheader("Frozen Animal Rule")
        st.dataframe(pd.json_normalize(id_rule), use_container_width=True)

    if history is not None and {"loss", "epoch"}.issubset(history.columns):
        y_cols = [c for c in ["loss", "val_loss"] if c in history.columns]
        fig = px.line(
            history,
            x="epoch",
            y=y_cols,
            markers=True,
            title="Training Loss by Epoch",
            labels={"value": "scaled MSE loss", "variable": "series"},
        )
        st.plotly_chart(fig, use_container_width=True)
    elif synthetic_metrics:
        st.info(
            f"Synthetic fallback trained for {synthetic_metrics['model_iterations']} iterations "
            "because saved TensorFlow training history was not found."
        )

    weights = weight_summary if weight_summary is not None else synthetic_weights
    if weights is not None and not weights.empty:
        st.subheader("Layer Weight Activity")
        st.caption(
            "This is not a biological interpretation by itself. It is an engineering view "
            "showing whether trained layers have nonzero learned weights and where weight "
            "magnitudes concentrate across the encoder, bottleneck, and decoder."
        )
        fig = px.imshow(
            weights[["mean_abs_weight", "max_abs_weight", "nonzero_weight_rate"]].T,
            x=weights["layer"],
            y=["mean abs", "max abs", "nonzero rate"],
            aspect="auto",
            title="Layer Weight Activity Summary",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(weights, use_container_width=True)
    else:
        st.caption(
            "Add `autoencoder_model.h5` in the artifact directory to show layer weight activity."
        )


def show_results_tab(
    rows: pd.DataFrame | None,
    id_summary: pd.DataFrame | None,
    contribution: pd.DataFrame | None,
    synthetic_scored: pd.DataFrame | None = None,
    synthetic_contribution: pd.DataFrame | None = None,
) -> None:
    st.subheader("3. AE Anomaly Detection Results")
    source = (
        "saved manuscript artifacts"
        if rows is not None or id_summary is not None
        else "synthetic fallback"
    )
    st.caption(f"Current source: {source}.")

    result_rows = rows if rows is not None else synthetic_scored
    result_contrib = contribution if contribution is not None else synthetic_contribution

    if id_summary is not None:
        flag_col = "is_positive_id" if "is_positive_id" in id_summary.columns else None
        if flag_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Animals", f"{len(id_summary):,}")
            c2.metric("Flagged Animals", f"{int(id_summary[flag_col].sum()):,}")
            c3.metric("Flagged Rate", f"{id_summary[flag_col].mean():.1%}")
        st.dataframe(id_summary, use_container_width=True)
        if {"id", "n_flags"}.issubset(id_summary.columns):
            fig = px.bar(
                id_summary.sort_values("n_flags", ascending=False),
                x="id",
                y="n_flags",
                color=flag_col if flag_col else None,
                title="Animal-Level Flag Counts",
            )
            st.plotly_chart(fig, use_container_width=True)
    elif synthetic_scored is not None:
        animal_flags = animal_level_flags(
            synthetic_scored,
            animal_col="animal_id",
            label_col="label",
            flag_col="row_flag",
            min_anomalous_fixes=3,
        )
        st.dataframe(animal_flags, use_container_width=True)

    if result_rows is not None and "reconstruction_error" in result_rows.columns:
        st.subheader("Row-Level Reconstruction Errors")
        color_col = "row_flag" if "row_flag" in result_rows.columns else None
        fig = px.histogram(
            result_rows,
            x="reconstruction_error",
            color=color_col,
            nbins=50,
            title="Row Reconstruction Error Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result_rows.head(500), use_container_width=True)

    if result_contrib is not None:
        st.subheader("Feature Residual Contributions")
        percent_cols = [c for c in result_contrib.columns if c.endswith("%") or c.endswith("_%")]
        if percent_cols:
            selected_col = st.selectbox("Contribution column", percent_cols)
            fig = px.bar(
                result_contrib.sort_values(selected_col, ascending=True),
                x=selected_col,
                y="feature",
                orientation="h",
                title=f"{selected_col} by Feature",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result_contrib, use_container_width=True)


def show_pipeline_runbook(profile_name: str, artifact_dir: Path) -> None:
    st.subheader("Run the Synthetic Manuscript Pipeline")
    st.write(
        "Use this when you want the dashboard to show a real artifact flow instead of the "
        "quick in-dashboard fallback. The command below generates synthetic train/test files, "
        "preprocesses them, trains the dense AE, applies the frozen animal rule, and writes "
        "feature residual contribution tables."
    )
    rel_output = artifact_dir.relative_to(ROOT) if artifact_dir.is_relative_to(ROOT) else artifact_dir
    st.code(
        "\n".join(
            [
                f"python scripts/run_synthetic_manuscript_pipeline.py \\",
                f"  --profile {profile_name} \\",
                f"  --output-dir {rel_output} \\",
                "  --epochs 40",
                "",
                "streamlit run dashboard/app.py",
            ]
        ),
        language="bash",
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "stage": "Preprocessing",
                    "script": "scripts/prepare_manuscript_features.py",
                    "artifacts": "train_processed_*.csv, val_processed_*.csv, test_processed_*.csv, encoder_*.pkl, scaler_*.pkl, feature_columns_*.pkl",
                    "dashboard tab": "1. Preprocessing",
                },
                {
                    "stage": "Training AE",
                    "script": "scripts/train_manuscript_autoencoder.py",
                    "artifacts": "autoencoder_model.h5, ae_ckpt.weights.h5, training_history.npy, reconstruction_threshold.pkl",
                    "dashboard tab": "2. AE Model",
                },
                {
                    "stage": "Results",
                    "script": "scripts/apply_manuscript_frozen_rule.py + scripts/manuscript_feature_contributions.py",
                    "artifacts": "row flags, residuals, ID summary, id_rule_params.pkl, feature contribution CSVs",
                    "dashboard tab": "3. AE Results",
                },
            ]
        ),
        use_container_width=True,
    )


def show_synthetic_metrics(
    scored: pd.DataFrame,
    sweep: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    st.subheader("Synthetic Demo Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensitivity", f"{metrics['sensitivity_recall']:.2f}")
    c2.metric("Specificity", f"{metrics['specificity']:.2f}")
    c3.metric("Precision", f"{metrics['precision']:.2f}")
    c4.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.2f}")

    fig = px.line(
        sweep,
        x="percentile",
        y=["flagged_fix_rate", "balanced_accuracy", "sensitivity", "specificity"],
        markers=True,
        title="Synthetic Threshold Sensitivity",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    cm = confusion_matrix(
        scored["label"].astype(int).to_numpy(),
        scored["row_flag"].astype(int).to_numpy(),
        labels=[0, 1],
    )
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual reference", "Actual comparison"],
            columns=["Predicted reference", "Predicted anomaly"],
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="CWD Telemetry Manuscript Pipeline", layout="wide")
    st.title("CWD Movement Anomaly Manuscript Pipeline")
    st.caption(
        "Presentation flow: preprocess synthetic telemetry, train a dense AE, then inspect "
        "preprocessing outputs, model artifacts, and anomaly-detection results."
    )

    with st.sidebar:
        st.header("Manuscript Artifacts")
        profile_name = st.selectbox(
            "Profile",
            ["wtd_primary", "wtd_four_pings", "mule_60_40"],
        )
        artifact_dir = Path(
            st.text_input(
                "Artifact directory",
                value=str(DEFAULT_ARTIFACT_DIR.relative_to(ROOT)),
                help=(
                    "Directory created by the manuscript scripts, such as outputs/wtd_primary. "
                    "The dashboard loads saved pkl, csv, npy, and h5 files from here."
                ),
            )
        )
        if not artifact_dir.is_absolute():
            artifact_dir = ROOT / artifact_dir

        use_synthetic = st.checkbox(
            "Use synthetic fallback when artifacts are missing",
            value=True,
        )
        threshold_percentile = st.slider(
            "Synthetic fallback threshold percentile",
            80,
            99,
            95,
        )

    paths = required_artifacts(profile_name, artifact_dir)
    exists = existing_artifacts(paths)
    artifact_ready = (
        exists["feature_columns"]
        and exists["train"]
        and exists["validation"]
        and exists["test"]
    )
    result_ready = exists["rows"] or exists["id_summary"] or exists["contrib"]

    status = artifact_status_table(paths, exists)
    with st.expander("Artifact Status", expanded=not artifact_ready):
        st.dataframe(status, use_container_width=True)

    feature_columns = load_optional_joblib(str(paths["feature_columns"]))
    threshold_info = load_optional_joblib(str(paths["threshold"]))
    id_rule = load_optional_joblib(str(paths["id_rule"]))
    history = load_training_history(str(paths["history"]))
    weight_summary = load_weight_summary(str(paths["model"]))

    train = load_optional_csv(str(paths["train"]))
    validation = load_optional_csv(str(paths["validation"]))
    test = load_optional_csv(str(paths["test"]))
    processed_inputs = [
        df.assign(split=name)
        for name, df in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]
        if df is not None
    ]
    processed = (
        pd.concat(processed_inputs, ignore_index=True)
        if processed_inputs
        else None
    )
    rows = load_optional_csv(str(paths["rows"]))
    id_summary = load_optional_csv(str(paths["id_summary"]))
    contribution = load_optional_csv(str(paths["contrib"]))

    synthetic_outputs = None
    if use_synthetic and (not artifact_ready or not result_ready):
        synthetic_df = load_synthetic_data(str(SAMPLE_PATH))
        try:
            synthetic_outputs = fit_synthetic_demo(synthetic_df, threshold_percentile)
        except ValueError as exc:
            st.warning(str(exc))

    if artifact_ready:
        st.success(
            "Loaded manuscript-profile processed files. The dashboard is reading saved local "
            "artifacts rather than fabricating the result surface."
        )
    elif use_synthetic:
        st.warning(
            "Manuscript artifacts were not found in that directory, so the dashboard is showing "
            "a quick in-dashboard fallback. For portfolio screenshots, run the synthetic "
            "manuscript pipeline so this dashboard loads real saved AE artifacts."
        )
    else:
        st.error("Required manuscript artifacts are missing and synthetic fallback is off.")

    synthetic_scored = None
    synthetic_contrib = None
    synthetic_sweep = None
    synthetic_weights = None
    synthetic_metrics = None
    if synthetic_outputs is not None:
        (
            synthetic_scored,
            synthetic_contrib,
            synthetic_sweep,
            synthetic_weights,
            synthetic_metrics,
        ) = synthetic_outputs

    tab_features, tab_model, tab_results, tab_run, tab_methods = st.tabs(
        [
            "1. Preprocessing",
            "2. AE Model",
            "3. AE Results",
            "Run Pipeline",
            "Methods",
        ]
    )

    with tab_features:
        show_feature_tab(profile_name, feature_columns, processed)

    with tab_model:
        show_model_tab(
            profile_name,
            threshold_info,
            id_rule,
            history,
            weight_summary,
            synthetic_metrics,
            synthetic_weights,
        )

    with tab_results:
        show_results_tab(rows, id_summary, contribution, synthetic_scored, synthetic_contrib)

    with tab_run:
        show_pipeline_runbook(profile_name, artifact_dir)
        st.subheader("Quick In-Dashboard Fallback Metrics")
        st.write(
            "This fallback exists only so the public repo can show something when no saved "
            "TensorFlow artifacts are present. The preferred portfolio path is the command above."
        )
        if synthetic_outputs is None:
            st.info("Synthetic fallback is disabled or could not run.")
        else:
            show_synthetic_metrics(synthetic_scored, synthetic_sweep, synthetic_metrics)

    with tab_methods:
        st.subheader("Dashboard Role")
        st.write(
            "The dashboard is deliberately secondary. The portfolio proof is the manuscript "
            "pipeline: preprocessing, saved feature order, autoencoder training, frozen "
            "thresholds, animal-level rules, and residual contribution tables."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "layer": "manuscript scripts",
                        "purpose": "faithful public-safe port of mule deer and WTD workflows",
                    },
                    {
                        "layer": "artifact viewer",
                        "purpose": (
                            "load local pkl/csv/npy/h5 outputs from synthetic or protected runs"
                        ),
                    },
                    {
                        "layer": "synthetic demo",
                        "purpose": "small fallback only for screenshots and public reproducibility",
                    },
                ]
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
