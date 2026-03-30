# Predicting Loan Default: Model Comparison Studying Feature Importance and Explainability

**Members:** Jiaming Liu, Aran Dharma

---

## Problem Description

We aim to predict whether a loan applicant will default on their loan, framed as a binary classification task. Beyond comparing model performance, we investigate two practical questions relevant to real-world deployment: (1) What is the tradeoff between model accuracy and interpretability? (2) Which data sources contribute most to predictive power?

Loan default prediction is important to responsible lending, with poor predictions leading to the rejection of worthy applicants or the approval of high-risk borrowers, which both have significant economic implications. Additionally, there are many adjacent tasks that involve approval classification on similar demographic data, and the results from this study could provide information on how those tasks could also be approached. In practice, lenders face constraints beyond raw accuracy. Regulations require explainable decisions and data collection has costs. We aim to quantify how much accuracy is sacrificed for interpretability, and we identify which auxiliary data sources justify their collection and maintenance costs through systematic ablation.

---

## Dataset

**Link:** https://www.kaggle.com/competitions/home-credit-default-risk/overview

**Description:** Loan application data from Home Credit, a consumer finance provider serving clients with limited credit history. The data includes a main application table joined with six auxiliary tables containing historical payment and credit behavior.

**Main Table:**
- `application_train.csv`: 307,511 loan applications (rows), 122 columns (features + TARGET label)
- `application_test.csv`: 48,744 rows, 121 columns (same features, no TARGET)

**Auxiliary Tables:** Relational history tables aggregated into applicant-level features via IDs (`SK_ID_CURR`, `SK_ID_BUREAU`):

| Table | Rows | Columns |
|-------|------|---------|
| `bureau` | 1,716,428 | 17 |
| `bureau_balance` | 27,299,925 | 3 |
| `previous_application` | 1,670,214 | 37 |
| `POS_CASH_balance` | 10,001,358 | 8 |
| `credit_card_balance` | 3,840,312 | 23 |
| `installments_payments` | 13,605,401 | 8 |

---

## Approach and Methodology

**Models:** Logistic Regression, Random Forest, Gradient Boosting / LightGBM, Neural Network

**Language and Packages:**
- Python 3.10+
- pandas, numpy, polars (data processing)
- scikit-learn (preprocessing, Logistic Regression, Random Forest, Gradient Boosting)
- PyTorch (neural network)
- SHAP (interpretability)
- matplotlib, seaborn (visualization)

**Preprocessing and Experimental Design:**
- Auxiliary tables are aggregated to the applicant level and merged into a shared tabular feature set.
- Ablation is performed over broad feature families rather than individual engineered columns, to keep the number of experiments manageable and the conclusions easier to interpret.
- The main ablation settings will include `application_only`, `full_features`, and `full_minus_<feature_family>` comparisons.
- The neural network will use the same aggregated tabular features as the other models rather than a separate sequence-based input representation.

**Evaluation and Metrics:**
- Stratified cross-validation with shared folds across models
- Classification metrics: accuracy, error, precision, recall, AUC-ROC, average precision
- Statistical comparison: paired t-tests or other paired tests on fold-level metrics, so reported differences between models are not based only on mean scores
- Ablation: Marginal AUC gain from grouped auxiliary feature families rather than individual engineered features
- Interpretability: SHAP feature importance rankings, with logistic regression used as a linear baseline rather than a primary importance tool

**Interpretability Notes:**
- Logistic regression is included as an interpretable linear baseline, but not as the main source of feature-importance claims.
- Because logistic-regression coefficients depend on feature scale, numeric features will be standardized before coefficient inspection.
- Coefficients will be used mainly for directional interpretation of linear effects, while SHAP will be the main tool for comparing global feature importance across models.

---

## Outcome

1. A performance comparison across four model families (Logistic Regression, Random Forest, Gradient Boosting / LightGBM, Neural Network), with boosting / neural models expected to lead on AUC and Logistic Regression providing a clear linear baseline.

2. An ablation study quantifying the marginal predictive value of each broad auxiliary data family: identifying which data sources contribute most to default prediction without requiring an impractical number of runs.

3. An interpretability analysis using SHAP, comparing whether different model families rely on the same features or surface different risk signals, while using logistic regression coefficients only for directional linear interpretation after scaling.

4. A practical recommendation balancing accuracy, interpretability, and data requirements for deployment in a lending context.

---

## Plan

| Jiaming Liu | Aran Dharma |
|-------------|-------------|
| EDA on main table | EDA on auxiliary tables |
| Feature engineering from auxiliary tables | Shared tabular preprocessing for all models |
| Logistic Regression + Random Forest | LightGBM + Neural Network |
| SHAP/interpretability analysis | Grouped ablation study (feature-family contribution) |
| Comparison analysis | |
