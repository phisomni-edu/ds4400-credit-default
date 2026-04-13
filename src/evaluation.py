"""Evaluation helpers for fold-level model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
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


def find_best_threshold(y_true, y_score, metric="f1"):
    y_true_series = pd.Series(y_true)
    y_score_series = pd.Series(y_score)

    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_value = -1.0

    for threshold in thresholds:
        metrics = evaluate_predictions(y_true_series, y_score_series, threshold=threshold)
        if metrics[metric] > best_value:
            best_value = metrics[metric]
            best_threshold = float(threshold)

    return best_threshold, best_value


def evaluate_with_best_f1_threshold(y_true, y_score):
    best_threshold, best_f1 = find_best_threshold(y_true, y_score, metric="f1")
    metrics = evaluate_predictions(y_true, y_score, threshold=best_threshold)
    metrics["threshold"] = best_threshold
    metrics["best_f1_from_sweep"] = best_f1
    return metrics


def build_threshold_table(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, y_score, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def find_threshold_for_target_precision(y_true, y_score, target_precision=0.30):
    threshold_table = build_threshold_table(y_true, y_score)
    eligible = threshold_table[threshold_table["precision"] >= target_precision].copy()

    if not eligible.empty:
        return (
            eligible.sort_values(["recall", "f1"], ascending=False)
            .iloc[0]
            .to_dict()
        )

    return threshold_table.sort_values("precision", ascending=False).iloc[0].to_dict()


def build_precision_recall_curve_df(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds,
        }
    )


def blend_probabilities(primary_scores, secondary_scores, primary_weight=0.5):
    primary_series = pd.Series(primary_scores).reset_index(drop=True)
    secondary_series = pd.Series(secondary_scores).reset_index(drop=True)
    return primary_weight * primary_series + (1 - primary_weight) * secondary_series


def make_fold_record(model_name, experiment_name, fold_index, metrics, extra_fields=None):
    record = {
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
