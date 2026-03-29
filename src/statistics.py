"""Statistical comparison helpers for cross-validated model results."""

import pandas as pd
from scipy import stats


def mean_confidence_interval(values, confidence=0.95):

    series = pd.Series(values, dtype=float).dropna()
    n = len(series)
    mean = float(series.mean())

    if n < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "sample_size": n}

    sem = stats.sem(series)
    margin = stats.t.ppf((1 + confidence) / 2, df=n - 1) * sem
    return {
        "mean": mean,
        "ci_low": float(mean - margin),
        "ci_high": float(mean + margin),
        "sample_size": n,
    }


def compare_paired_models(results_df, left_model, right_model, metric="roc_auc", experiment=None):

    filtered = results_df.copy()
    if experiment is not None:
        filtered = filtered.loc[filtered["experiment"] == experiment]

    paired = (
        filtered.loc[filtered["model"].isin([left_model, right_model]), ["fold", "model", metric]]
        .pivot(index="fold", columns="model", values=metric)
        .dropna()
    )

    left = paired[left_model]
    right = paired[right_model]
    differences = left - right
    n = int(len(differences))

    t_stat, t_pvalue = stats.ttest_rel(left, right)
    w_stat, w_pvalue = stats.wilcoxon(left, right, zero_method="wilcox")

    std = float(differences.std(ddof=1))
    effect_size = float(differences.mean() / std) if std else 0.0
    mean_delta = float(differences.mean())
    ci = mean_confidence_interval(differences)

    return {
        "left_model": left_model,
        "right_model": right_model,
        "experiment": experiment or "all",
        "metric": metric,
        "n_folds": n,
        "mean_delta": mean_delta,
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_pvalue),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "cohens_d": effect_size,
        "delta_ci_low": float(ci["ci_low"]),
        "delta_ci_high": float(ci["ci_high"]),
    }


def compare_all_model_pairs(results_df, metric="roc_auc", experiment=None):
    filtered = results_df if experiment is None else results_df.loc[results_df["experiment"] == experiment]
    models = sorted(filtered["model"].dropna().unique().tolist())

    rows = []
    for index, left_model in enumerate(models):
        for right_model in models[index + 1 :]:
            rows.append(
                compare_paired_models(
                    results_df=filtered,
                    left_model=left_model,
                    right_model=right_model,
                    metric=metric,
                    experiment=experiment,
                )
            )

    return pd.DataFrame(rows)
