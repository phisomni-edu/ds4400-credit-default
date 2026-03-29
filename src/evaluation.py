"""Evaluation helpers for fold-level model comparison."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_score, threshold=0.5):
    y_true_series = pd.Series(y_true)
    y_score_series = pd.Series(y_score)
    y_pred = (y_score_series >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true_series, y_pred),
        "precision": precision_score(y_true_series, y_pred, zero_division=0),
        "recall": recall_score(y_true_series, y_pred, zero_division=0),
        "f1": f1_score(y_true_series, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true_series, y_score_series),
        "average_precision": average_precision_score(y_true_series, y_score_series),
    }


def make_fold_record(model_name, experiment_name, fold_index, metrics, extra_fields=None):
    record={
        "model": model_name,
        "experiment": experiment_name,
        "fold": fold_index,
        **metrics,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def summarize_cv_results(results_df, group_cols=None):
    if group_cols is None:
        group_cols = ["experiment", "model"]

    metric_cols = [
        column
        for column in results_df.columns
        if column not in set(group_cols + ["fold"])
        and pd.api.types.is_numeric_dtype(results_df[column])
    ]

    summary = results_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary
