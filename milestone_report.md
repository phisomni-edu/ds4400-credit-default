# Project Milestone Report

**Project:** Predicting Loan Default: Model Comparison, Feature Importance, and Explainability  
**Team Members:** Jiaming Liu, Aran Dharma

---

## Problem Description

This project studies loan default prediction using the Home Credit Default Risk dataset. The machine learning task is a **binary classification** problem: given an applicant and related credit history, predict whether the applicant will default on a loan (`TARGET = 1`) or not (`TARGET = 0`).

This problem is important because credit-risk models directly influence lending decisions. Poor predictions may reject qualified applicants, approve high-risk borrowers, or rely on patterns that are difficult to justify in regulated settings. Our project focuses on three related questions:

1. How much predictive performance is gained by moving from simple linear models to more complex nonlinear models?
2. Which auxiliary data sources contribute the most useful signal?
3. How much interpretability is lost when stronger models are used?

The project has been informed by the Home Credit Kaggle competition overview, the SHAP paper by Lundberg and Lee (2017), the LightGBM paper by Ke et al. (2017), and scikit-learn documentation for logistic regression. These references helped shape the current methodology: grouped ablation rather than feature-by-feature ablation, SHAP as the main interpretability method for nonlinear models, and fold-level statistical comparison rather than reliance on a single score.

---

## Dataset

The project uses the **Home Credit Default Risk** dataset from Kaggle. The dataset includes one main application table and several auxiliary relational tables containing credit history, previous applications, payment records, and account balances.

### Basic Dataset Description

- `application_train.csv`: 307,511 rows, 122 columns
- `application_test.csv`: 48,744 rows, 121 columns

Auxiliary tables include:

- `bureau`
- `bureau_balance`
- `previous_application`
- `POS_CASH_balance`
- `credit_card_balance`
- `installments_payments`

The training data is imbalanced:

- Non-default: 282,686
- Default: 24,825
- Default rate: 8.07%

### Feature Insights

The raw data contains a mix of numeric, binary, categorical, and identifier fields. Several housing-related features have substantial missingness, so the current pipeline uses simple imputation rather than row deletion.

Because the auxiliary tables contain many repeated records per applicant, they cannot be used directly in standard tabular models. We therefore aggregate them to the applicant level before training. This aggregation produced:

- `train_features.parquet`: 307,511 x 471
- `test_features.parquet`: 48,744 x 470

The resulting feature set includes:

- application-derived features
- bureau-related summaries
- previous-application summaries
- POS cash summaries
- credit-card summaries
- installment-payment summaries

Early EDA showed that the auxiliary tables vary greatly in repetition per applicant, which supports the use of grouped aggregation and later grouped ablation.

### Exploratory Data Analysis

EDA completed so far includes:

- main-table inspection of feature types, class balance, missingness, and selected variable distributions
- auxiliary-table inspection of table sizes, records per applicant, missingness, and key status distributions

These analyses were used primarily to guide feature engineering and preprocessing. They also confirmed that class imbalance and relational structure are two central characteristics of this dataset.

---

## Approach and Methodology

The overall approach is to convert the relational dataset into a shared applicant-level tabular dataset, train several model families on the same feature base, and compare them with respect to performance and interpretability.

### Feature Engineering and Preprocessing

The current pipeline performs the following steps:

1. Clean sentinel values such as `DAYS_EMPLOYED = 365243`
2. Create simple derived application features, such as ratio-based financial variables
3. Aggregate each auxiliary table to the applicant level
4. Join all applicant-level features into a shared processed dataset

Two model-ready datasets were then created:

- `linear_ready` for Logistic Regression and the neural network
- `tree_ready` for Random Forest and LightGBM

The current preprocessing includes:

- removal of identifier columns from model inputs
- dropping constant columns
- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding of categorical variables
- numeric standardization for the linear-ready dataset

Saved modeling datasets currently have the following shapes:

- `train_linear_ready.parquet`: 307,511 x 600
- `test_linear_ready.parquet`: 48,744 x 599
- `train_tree_ready.parquet`: 307,511 x 600
- `test_tree_ready.parquet`: 48,744 x 599

### Models and Experimental Plan

The full project plan still includes four model families:

- Logistic Regression
- Random Forest
- LightGBM
- Neural Network

Based on instructor feedback, the ablation strategy was revised from individual-feature ablation to **grouped feature-family ablation**. The planned comparisons include `application_only`, `full_features`, and `full_minus_<feature_family>` settings. This approach is more computationally feasible and easier to interpret.

Model comparisons will ultimately use shared folds and paired statistical tests. Logistic regression is retained as a linear baseline, but SHAP will be the primary interpretability method for stronger nonlinear models.

### Preliminary Model Results

An initial baseline notebook has already been completed using a single stratified train/validation split. Two models have been trained so far:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
|-------|---------:|----------:|-------:|---:|--------:|---------------:|
| Logistic Regression | 0.7110 | 0.1755 | 0.6973 | 0.2804 | 0.7752 | 0.2635 |
| Random Forest | 0.8931 | 0.3069 | 0.2578 | 0.2802 | 0.7664 | 0.2447 |

These results are preliminary because they are based on a single validation split rather than cross-validation. However, they already show a meaningful tradeoff. Logistic Regression currently performs better on ROC-AUC, average precision, and recall, while Random Forest attains higher accuracy and precision but identifies substantially fewer positive default cases. Since the dataset is imbalanced, the ranking and retrieval metrics are more informative than accuracy alone.

### Challenges and Changes from the Proposal

Several changes were made after beginning implementation:

- The relational structure of the dataset required applicant-level aggregation before modeling.
- The scale of the auxiliary tables made efficient preprocessing necessary.
- Logistic-regression coefficients are no longer treated as the main feature-importance method.
- Ablation was revised to grouped feature families.
- Statistical testing was added to the evaluation plan.

Overall, the project has progressed from an initial proposal to a functioning data pipeline with baseline modeling results.

---

## Remaining Work

The remaining work is concentrated in the experimental and reporting stages:

1. Add LightGBM and a first neural-network baseline.
2. Replace the single train/validation split with cross-validated evaluation.
3. Run grouped ablation experiments over the main feature families.
4. Perform statistical comparisons on fold-level metrics.
5. Complete SHAP-based interpretability analysis.
6. Finalize the report and presentation.

---

## Team Member Contribution

### Jiaming Liu

Current contributions:

- main-table EDA direction
- project framing and proposal development
- baseline modeling work for Logistic Regression and Random Forest

Planned remaining work:

- extend evaluation for Logistic Regression and Random Forest
- contribute to comparison analysis and final report writing

### Aran Dharma

Current contributions:

- auxiliary-table EDA
- feature engineering from auxiliary tables
- preprocessing pipeline for model-ready datasets
- setup for grouped ablation and statistical-comparison workflow

Planned remaining work:

- implement LightGBM and neural-network baselines
- run grouped ablation experiments
- support interpretability analysis and final integration

---

## Milestone Summary

At this milestone, the project has established the core data pipeline, completed substantial EDA, generated applicant-level engineered features, built model-ready datasets, and trained two initial baseline models. The main remaining tasks are broader model comparison, cross-validated evaluation, grouped ablation, and interpretability analysis.
