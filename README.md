# Customer Churn Prediction

An end-to-end machine learning system that predicts whether a telecom customer is likely to churn, using the IBM Telco Customer Churn dataset. Built as a portfolio project demonstrating the full ML workflow: data cleaning, EDA, feature engineering, model comparison, hyperparameter tuning, explainability (SHAP), and deployment (Streamlit).

## Overview

Customer churn — when a customer stops using a company's service — is one of the most common and costly problems in subscription-based businesses. This project builds a classification model that flags customers at elevated risk of churning, so a business could prioritize retention efforts.

## Problem Statement

> Can historical customer data (demographics, account details, services subscribed, billing information) be used to identify customers who are at higher risk of churn?

## Dataset

**Source:** IBM Telco Customer Churn dataset (7,043 customers, 21 columns).

**Target:** `Churn` (Yes/No) — imbalanced at ~26.5% churn / 73.5% retained.

**Known data quality issue found during inspection:** `TotalCharges` is stored as text and contains 11 blank values, all belonging to customers with `tenure == 0` (brand-new customers with no billing history yet). These were converted to numeric and imputed to `0.0` — the factually correct value, not an estimate.

## Objectives

- Build a reliable churn classifier evaluated on metrics appropriate for an imbalanced target (not accuracy alone)
- Compare multiple model families and justify the final choice
- Make the model's predictions explainable at both a global and individual-customer level
- Ship a usable prediction interface (Streamlit)

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, Joblib, Streamlit

## Project Workflow

```text
Data → EDA → Cleaning → Feature Engineering → Preprocessing →
Model Training → Model Comparison → Hyperparameter Tuning →
Evaluation → SHAP Explainability → Streamlit Deployment
```

## Results

All numbers below are from actual experiments on a held-out 20% test set (1,409 customers), stratified to preserve the true churn rate.

### Baseline model comparison (5-fold cross-validated ROC-AUC + test set metrics)

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC | CV ROC-AUC (mean ± std) |
|---------------------|----------|-----------|--------|--------|---------|--------------------------|
| Logistic Regression | 0.7984   | 0.6562    | 0.5053 | 0.5710 | 0.8451  | 0.8473 ± 0.0112          |
| Random Forest       | 0.7857   | 0.6208    | 0.4947 | 0.5506 | 0.8190  | 0.8244 ± 0.0115          |
| XGBoost (default)   | 0.7814   | 0.6051    | 0.5080 | 0.5523 | 0.8182  | 0.8236 ± 0.0112          |

**Model selection note:** Logistic Regression slightly outperforms the default tree ensembles on ROC-AUC out of the box, which is a legitimate and interesting result — not every problem needs a complex model, and interpretable coefficients are a real advantage for a churn use case that a business analyst may want to audit. XGBoost was selected for tuning because it has the most room to improve via hyperparameter search and typically overtakes linear models once properly regularized on tabular data.

### Final tuned model: XGBoost (RandomizedSearchCV, 5-fold CV, 40 iterations)

Best hyperparameters found: `n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.9, reg_lambda=2.0`

| Metric | Value |
|---|---|
| Test Accuracy | 0.8055 |
| Test Precision | 0.6736 |
| Test Recall | 0.5187 |
| Test F1 | 0.5861 |
| Test ROC-AUC | **0.8487** |
| CV ROC-AUC | 0.8499 |

Confusion matrix (test set, threshold = 0.5):

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 941 | 94 |
| **Actual: Churn** | 180 | 194 |

Tuning improved ROC-AUC from 0.8182 (default XGBoost) to 0.8487, and lifted precision on the churn class from 0.605 to 0.674 — a meaningful gain in how trustworthy a "flagged as high risk" prediction is.

### Threshold analysis

The default 0.5 threshold is not necessarily the best operating point for a business. Lower thresholds catch more actual churners (higher recall) at the cost of more false alarms (lower precision):

| Threshold | Precision | Recall | F1 | Customers Flagged |
|---|---|---|---|---|
| 0.30 | 0.537 | 0.783 | 0.637 | 546 |
| 0.40 | 0.601 | 0.666 | 0.632 | 414 |
| 0.50 | 0.674 | 0.519 | 0.586 | 288 |
| 0.60 | 0.743 | 0.340 | 0.466 | 171 |
| 0.70 | 0.812 | 0.219 | 0.345 | 101 |

**Business interpretation:** if the cost of missing a genuine churner is high relative to the cost of an unnecessary retention offer, a lower threshold (e.g. 0.30–0.40) is preferable even though it flags more customers overall.

## Explainability

**Feature importance (XGBoost, gain-based):** the top drivers are, in order — having a month-to-month contract (by a wide margin), fiber optic internet service, no online security add-on, and no tech support add-on. See `reports/feature_importance.csv` for the full ranking.

**SHAP analysis:** global summary plots (`reports/figures/shap_global_importance.png`, `shap_beeswarm.png`) confirm and add direction to the feature importance ranking — e.g. showing that having a month-to-month contract pushes predictions toward churn, while long tenure pushes them away. A local explanation for an individual high-risk customer is saved at `reports/figures/shap_local_high_risk_customer.png`.

**Important caveat:** feature importance and SHAP values show *association*, not *causation*. A feature ranking highly does not mean changing that feature would change a real customer's behavior — only that it was informative for this dataset's historical patterns.

## Application

A Streamlit app (`app/app.py`) lets you enter a customer's details and get:
- Churn probability
- Risk level (Low / Medium / High)
- A plain-language prediction
- The top model-driven factors behind that specific prediction

### Running the app

```bash
streamlit run app/app.py
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Usage

```bash
# 1. Clean the raw data
python src/data_preprocessing.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Train, compare, and tune models
python src/train.py

# 4. Threshold analysis + feature importance
python src/evaluate.py

# 5. SHAP explainability
python src/explainability.py

# 6. Predict for a single example customer
python src/predict.py

# 7. Launch the app
streamlit run app/app.py
```

## Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original, untouched CSV
│   └── processed/               # Cleaned + feature-engineered data, train/test splits
├── src/
│   ├── data_preprocessing.py    # Loading + cleaning
│   ├── feature_engineering.py   # Derived features
│   ├── train.py                 # Split, pipeline, model comparison, tuning
│   ├── evaluate.py              # Threshold analysis, feature importance
│   ├── explainability.py        # SHAP
│   └── predict.py               # Single-customer prediction function
├── models/                      # Saved trained pipelines (joblib)
├── app/
│   └── app.py                   # Streamlit application
├── reports/
│   ├── figures/                 # SHAP plots
│   ├── model_comparison.csv
│   ├── final_results.json
│   ├── threshold_analysis.csv
│   └── feature_importance.csv
├── requirements.txt
└── README.md
```

## Limitations

- **Dataset size and origin:** 7,043 customers from a single (IBM sample) telecom provider; findings may not generalize to other companies or markets.
- **Class imbalance:** ~26.5% churn rate. Addressed via stratified splitting and threshold-independent evaluation (ROC-AUC), but recall on the minority class (52%) still leaves roughly half of actual churners undetected at the default threshold.
- **No causal claims:** the model identifies statistical associations, not causal drivers. "Month-to-month contract" being the top feature does not prove that switching a customer to a longer contract would prevent their churn.
- **Historical data only:** the model reflects patterns in past behavior and does not account for future market changes, competitor actions, or shifts in customer expectations.
- **Static model:** in a real deployment this model would need monitoring for data drift and periodic retraining.
- **Threshold is a business decision, not fixed by this project:** the default 0.5 threshold used in headline metrics is a reasonable default, not necessarily optimal for any specific business's cost structure.

## Future Improvements

- Add SMOTE or class-weighting experiments to see if minority-class recall can be improved without sacrificing too much precision
- Deploy with a monitoring layer to track prediction drift over time
- Expand feature engineering with customer interaction/support-ticket data, if available
- A/B test retention interventions against model-flagged high-risk customers to validate real-world impact

## Resume-Ready Project Description

> **Customer Churn Prediction** — Built an end-to-end ML pipeline (Python, scikit-learn, XGBoost) to predict customer churn on the IBM Telco dataset (7,043 records), comparing Logistic Regression, Random Forest, and XGBoost with 5-fold cross-validation. Tuned the final XGBoost model via RandomizedSearchCV, achieving 0.849 test ROC-AUC (up from 0.818 baseline). Applied SHAP for global and local model explainability, and deployed an interactive Streamlit application for individual customer risk scoring.
