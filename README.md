# Home Credit Default

Course project for predicting loan default risk on the Home Credit Default Risk Kaggle dataset.

## Project Goals

- Compare logistic regression, random forest, gradient boosting / LightGBM, and neural network baselines.
- Measure the marginal value of each auxiliary table through ablation.
- Analyze feature importance and explainability with SHAP.

## Repository Layout

```text
home-credit-default/
├── configs/
├── data/
├── models/
├── notebooks/
├── outputs/
├── reports/
└── src/
```

`reports/proposal.md` contains the original project proposal.

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

## Next Steps

- Implement raw data loaders in `src/data_loading.py`.
- Build table aggregations in `src/feature_engineering.py`.
- Add preprocessing, training, evaluation, and SHAP pipelines.
