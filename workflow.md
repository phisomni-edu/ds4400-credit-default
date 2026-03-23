# Project Workflow Guide

## Phase 1: Setup (Week 1)

### 1.1 Repository Setup
```bash
# Create repo
git init home-credit-default
cd home-credit-default

# Create directory structure
mkdir -p data/raw data/processed notebooks src configs models outputs/figures outputs/tables outputs/logs reports

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install pandas numpy polars scikit-learn lightgbm torch shap matplotlib seaborn jupyter pyyaml pyarrow
pip freeze > requirements.txt
```

### 1.2 Data Download
- Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
- Place all CSVs in `data/raw/`
- Add `data/` to `.gitignore` (files are too large for Git)

### 1.3 .gitignore
```
data/
models/
outputs/logs/
.ipynb_checkpoints/
__pycache__/
*.pyc
venv/
.DS_Store
```

---

## Phase 2: EDA (Week 1-2)

### Person A: Main Table (`01_eda_main_table.ipynb`)
- Load `application_train.csv`
- Examine target distribution (class imbalance)
- Missing value analysis
- Feature distributions and correlations
- Identify categorical vs. numerical features

### Person B: Auxiliary Tables (`02_eda_auxiliary_tables.ipynb`)
- Load each auxiliary table
- Understand schema and relationships (SK_ID_CURR, SK_ID_BUREAU, etc.)
- Row counts per application
- Identify useful aggregation targets

### Sync Point
- Share findings
- Agree on feature engineering strategy

---

## Phase 3: Feature Engineering (Week 3-4)

### Build `src/feature_engineering.py`
```python
def aggregate_bureau(bureau_df, bureau_balance_df):
    """Aggregate credit bureau data per application."""
    # Count of prior credits
    # Average/max credit amount
    # Delinquency counts
    # ...
    return bureau_features

def aggregate_previous_apps(prev_app_df):
    """Aggregate previous Home Credit applications."""
    return prev_app_features

def aggregate_payments(installments_df, pos_cash_df, cc_balance_df):
    """Aggregate payment behavior."""
    return payment_features

def build_full_feature_set(app_df, bureau_features, prev_features, payment_features):
    """Merge all features into final training set."""
    return final_df
```

### Run in `03_feature_engineering.ipynb`
- Execute aggregation functions
- Merge into single DataFrame
- Save to `data/processed/train_features.parquet`
- Document features in `feature_definitions.json`

### Task Split
| Person A | Person B |
|----------|----------|
| `aggregate_bureau()` | `aggregate_payments()` |
| `aggregate_previous_apps()` | Merge and validation |

---

## Phase 4: Model Training (Week 5-6)

### Build `src/models.py`
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import torch.nn as nn

def train_logistic_regression(X_train, y_train, config):
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, config):
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, config):
    model = lgb.LGBMClassifier(**config)
    model.fit(X_train, y_train)
    return model

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        # Define layers
        pass
    
    def forward(self, x):
        pass

def train_neural_net(X_train, y_train, config):
    # Training loop
    pass
```

### Run in `04_model_training.ipynb`
- Load processed features
- Train/validation split (stratified)
- Train all 4 models
- Evaluate on validation set
- Save models to `models/`
- Log metrics to `outputs/tables/model_metrics.csv`

### Task Split
| Person A | Person B |
|----------|----------|
| Logistic Regression | LightGBM |
| Random Forest | Neural Network |

---

## Phase 5: Ablation Study (Week 7)

### Configure `configs/ablation_configs.yaml`
```yaml
experiments:
  baseline:
    features: ["main_only"]
  
  plus_bureau:
    features: ["main_only", "bureau"]
  
  plus_payments:
    features: ["main_only", "payments"]
  
  plus_previous:
    features: ["main_only", "previous_apps"]
  
  full:
    features: ["main_only", "bureau", "payments", "previous_apps"]
```

### Run in `05_ablation_study.ipynb`
- For each experiment config:
  - Select appropriate feature columns
  - Train all 4 models
  - Record AUC
- Compute marginal AUC gain per data source
- Generate ablation plot
- Save results to `outputs/tables/ablation_auc.csv`

---

## Phase 6: SHAP Analysis (Week 7)

### Run in `06_shap_analysis.ipynb`
```python
import shap

# LightGBM (fast)
explainer_lgbm = shap.TreeExplainer(lgbm_model)
shap_values_lgbm = explainer_lgbm.shap_values(X_val)
shap.summary_plot(shap_values_lgbm, X_val, show=False)
plt.savefig('outputs/figures/shap_summary_lgbm.png')

# Logistic Regression (coefficients)
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# Compare top features across models
# ...
```

### Outputs
- SHAP summary plots per model
- Top-20 feature comparison table
- Analysis of agreement/disagreement across models

---

## Phase 7: Report & Presentation (Week 8)

### Final Report Structure
1. Introduction & Problem Statement
2. Dataset Description
3. Feature Engineering Approach
4. Model Comparison Results
5. Ablation Study Findings
6. Interpretability Analysis
7. Conclusions & Recommendations

### Figures to Include
- Model comparison bar chart (AUC by model)
- Ablation study plot (cumulative AUC by data source)
- SHAP summary plots
- Feature importance comparison table

---

## Git Workflow

### Branching Strategy (Simple)
```bash
main              # Stable, working code
├── feature/eda
├── feature/feature-engineering  
├── feature/models
├── feature/ablation
└── feature/shap
```

### Collaboration Pattern
1. Each person works on their own branch
2. Push frequently
3. Create pull request when task is complete
4. Review each other's code briefly
5. Merge to main

### Commit Message Convention
```
[phase] brief description

Examples:
[eda] add missing value analysis for main table
[features] implement bureau aggregation functions
[models] add LightGBM training with class weights
[ablation] run all experiments and save results
```

---

## Checkpoints & Sync Points

| Week | Checkpoint | Sync |
|------|------------|------|
| 2 | EDA complete | Share findings, agree on features |
| 4 | Features engineered, saved to parquet | Validate merged dataset together |
| 6 | All models trained | Compare metrics, debug issues |
| 7 | Ablation + SHAP complete | Review findings together |
| 8 | Report draft | Edit together, prepare presentation |