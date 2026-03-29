# Home Credit Default

Course project for predicting loan default risk on the Home Credit Default Risk Kaggle dataset.

## Project Goals

- Compare logistic regression, random forest, gradient boosting / LightGBM, and neural network baselines.
- Measure the marginal value of broad feature families through grouped ablation.
- Evaluate model differences with fold-level metrics and paired statistical tests.
- Analyze feature importance and explainability with SHAP, while treating logistic regression as a linear baseline rather than a standalone importance tool.

## Repository Layout

```text
home-credit-default/
|- configs/
|- data/
|- models/
|- notebooks/
|- outputs/
|- reports/
`- src/
```

`reports/proposal.md` contains the current proposal.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place the Kaggle CSV files in `data/raw/`:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`

Generated features should be written to `data/processed/`. Trained models belong in `models/`, and plots / tables / logs belong in `outputs/`.

## Experiment Strategy

- Run grouped ablations over major source families instead of ablating individual engineered columns.
- Use the same stratified cross-validation splits for every model and ablation experiment.
- Save fold-level metrics so we can run paired t-tests or Wilcoxon signed-rank tests on model comparisons.
- Standardize inputs for logistic regression and use its coefficients only for directional interpretation of linear effects.

## Next Steps

- Implement raw data loaders in `src/data_loading.py`.
- Build table aggregations in `src/feature_engineering.py`.
- Use `configs/ablation_configs.yaml` and `configs/experiment_configs.yaml` to drive experiments.
- Use `src/preprocessing.py`, `src/evaluation.py`, and `src/statistics.py` for reproducible fold-level comparison.
