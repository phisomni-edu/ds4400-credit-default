# Project Workflow Guide

## Phase 1: Setup

### 1.1 Repository Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1.2 Data Download
- Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
- Place all CSVs in `data/raw/`
- Keep raw data, processed features, trained models, and generated outputs out of git

## Phase 2: EDA

### `01_eda_main_table.ipynb`
- Load `application_train.csv`
- Examine target imbalance
- Review missingness, feature types, and basic distributions

### `02_eda_auxiliary_tables.ipynb`
- Load each auxiliary table
- Understand join keys and row multiplicity
- Identify aggregation candidates for grouped feature families

## Phase 3: Feature Engineering

### `src/feature_engineering.py`
- Build aggregations for each auxiliary source
- Keep engineered columns tagged by source family so grouped ablation is straightforward
- Save merged training and test features to `data/processed/`

## Phase 4: Model Training

### `04_model_training.ipynb`
- Load processed features
- Build shared stratified cross-validation folds
- Train all models on the same folds
- Save fold-level metrics to `outputs/tables/cv_metrics_by_fold.csv`
- Save fitted models to `models/`

### Modeling Notes
- Logistic regression is a linear baseline, not the main importance method
- Logistic regression inputs should be standardized before coefficient inspection
- Tree-based and boosting models should use SHAP or permutation importance for feature analysis

## Phase 5: Grouped Ablation

### `configs/ablation_configs.yaml`
- Define experiments over broad feature families rather than individual features
- Prioritize `application_only`, `full_features`, and `full_minus_<group>` runs

### `05_ablation_study.ipynb`
- Select grouped feature families per experiment
- Reuse the same CV folds as the main model comparison
- Record fold-level metrics for each experiment
- Save aggregated ablation outputs to `outputs/tables/ablation_results.csv`

## Phase 6: Statistical Comparison

### `07_statistical_comparison.ipynb`
- Compare models on shared fold results
- Run paired t-tests and Wilcoxon signed-rank tests where appropriate
- Report confidence intervals for fold-level deltas
- Save outputs to `outputs/tables/statistical_tests.csv`

## Phase 7: Interpretability

### `06_shap_analysis.ipynb`
- Use SHAP as the primary importance method for nonlinear models
- Use logistic regression coefficients only for directional linear interpretation after scaling
- Compare whether top signals are consistent across model families

## Phase 8: Final Deliverables

### Report
1. Introduction and problem framing
2. Dataset and grouped feature families
3. Feature engineering pipeline
4. Cross-validated model comparison
5. Grouped ablation results
6. Statistical significance analysis
7. SHAP-based interpretability findings
8. Conclusion and deployment recommendation
