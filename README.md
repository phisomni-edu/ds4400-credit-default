# Home Credit Default Risk

Course project on predicting loan default risk with the Home Credit Default Risk Kaggle dataset.

## Project Summary

This project treats credit default prediction as a binary classification problem. The main workflow is:

1. load the raw relational Kaggle tables
2. aggregate auxiliary credit-history tables to the applicant level
3. build model-ready tabular datasets
4. train and compare multiple baseline models
5. run grouped ablation and SHAP-based interpretability analysis

The strongest standalone model was LightGBM, and the best overall model was a weighted hybrid of LightGBM and Logistic Regression.

## Repository Layout

```text
home-credit-default/
|- configs/
|- data/
|  |- raw/
|  `- processed/
|- notebooks/
|- outputs/
|- src/
|- final_report.md
|- milestone_report.md
|- proposal.md
`- README.md
```

Key notebooks:

- `notebooks/01_eda_main_table.ipynb`
- `notebooks/02_eda_auxiliary_tables.ipynb`
- `notebooks/04_model_training.ipynb`
- `notebooks/05_ablation_study.ipynb`
- `notebooks/06_shap_analysis.ipynb`

Key source files:

- `src/data_loading.py`
- `src/feature_engineering.py`
- `src/preprocessing.py`
- `src/models.py`
- `src/evaluation.py`

## Setup

Install dependencies:

```bash
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

Raw and processed data are ignored by git.

## Reproducing the Pipeline

### 1. Build applicant-level engineered features

```bash
python src/feature_engineering.py
```

This writes:

- `data/processed/train_features.parquet`
- `data/processed/test_features.parquet`
- `data/processed/feature_definitions.json`

### 2. Build model-ready datasets

```bash
python src/preprocessing.py
```

This writes:

- `data/processed/train_linear_ready.parquet`
- `data/processed/test_linear_ready.parquet`
- `data/processed/train_tree_ready.parquet`
- `data/processed/test_tree_ready.parquet`
- `data/processed/preprocessing_summary.json`

### 3. Run experiments

Open and run:

- `notebooks/04_model_training.ipynb`
- `notebooks/05_ablation_study.ipynb`
- `notebooks/06_shap_analysis.ipynb`

These notebooks produce the final model-comparison tables, ablation results, and SHAP outputs under `outputs/`.

## Saved Outputs

Main result tables:

- `outputs/tables/model_metrics_default.csv`
- `outputs/tables/model_metrics_tuned.csv`
- `outputs/tables/hybrid_weight_sweep.csv`
- `outputs/tables/ablation_results.csv`
- `outputs/tables/shap_top_features.csv`

Main figures:

- `outputs/figures/model_comparison.png`
- `outputs/figures/precision_recall_curves.png`
- `outputs/figures/ablation_results.png`
- `outputs/figures/shap_summary_lgbm.png`

## Notes

- Logistic Regression is used as a linear baseline.
- LightGBM is the strongest standalone model.
- The weighted `LightGBM + Logistic Regression` hybrid is the main novel contribution.
- The stacked ensemble was exploratory and did not outperform the main models.
