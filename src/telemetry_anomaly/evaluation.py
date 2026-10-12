"""Evaluation utilities for row-level and animal-level anomaly results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def row_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute row-level classification metrics for GPS fixes."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def animal_level_flags(
    df: pd.DataFrame,
    *,
    animal_col: str = "animal_id",
    label_col: str = "label",
    flag_col: str = "is_anomaly",
    min_anomalous_fixes: int = 1,
) -> pd.DataFrame:
    """Aggregate row flags to animal-level calls."""

    grouped = df.groupby(animal_col)
    out = grouped.agg(
        true_label=(label_col, "max"),
        n_fixes=(flag_col, "size"),
        n_anomalous_fixes=(flag_col, "sum"),
    ).reset_index()
    out["predicted_label"] = (out["n_anomalous_fixes"] >= min_anomalous_fixes).astype(int)
    return out
