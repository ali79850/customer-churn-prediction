"""
Train/test split, preprocessing pipeline, model training, comparison,
cross-validation, and hyperparameter tuning for churn prediction.

Design decisions:
- Stratified 80/20 split: Churn is imbalanced (~73.5%/26.5%), stratification
  keeps that ratio consistent between train and test.
- Preprocessing pipeline (imputation/scaling/encoding) is fit ONLY on the
  training fold inside each CV split via sklearn Pipeline - this prevents
  data leakage from test statistics into training.
- Primary tuning/selection metric: ROC-AUC (threshold-independent, robust
  under class imbalance). We report full metrics for all models regardless.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="if_binary")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    return metrics, cm, report


def main():
    df = pd.read_csv("data/processed/telco_features.csv")
    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                     "ServiceCount", "AvgMonthlySpend"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"\n{name} 5-fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        metrics, cm, report = evaluate_model(name, pipe, X_test, y_test)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)
        results.append(metrics)
        print(f"{name} test set metrics: {metrics}")
        print(f"Confusion matrix:\n{np.array(cm)}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== MODEL COMPARISON (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("reports/model_comparison.csv", index=False)

    # --- Hyperparameter tuning on the best baseline model: XGBoost ---
    print("\n=== Hyperparameter tuning: XGBoost (RandomizedSearchCV) ===")
    xgb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
    ])

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    search = RandomizedSearchCV(
        xgb_pipe, param_distributions=param_dist, n_iter=40,
        scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_
    tuned_metrics, tuned_cm, tuned_report = evaluate_model(
        "XGBoost_Tuned", best_model, X_test, y_test
    )
    print(f"\nTuned XGBoost test metrics: {tuned_metrics}")
    print(f"Confusion matrix:\n{np.array(tuned_cm)}")
    print(f"\nClassification report:\n{tuned_report}")

    # Save everything needed for later phases
    joblib.dump(best_model, "models/churn_model.joblib")
    joblib.dump(fitted_pipelines["LogisticRegression"], "models/logreg_model.joblib")
    joblib.dump(fitted_pipelines["RandomForest"], "models/rf_model.joblib")
    joblib.dump(fitted_pipelines["XGBoost"], "models/xgb_baseline_model.joblib")

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    final_report = {
        "baseline_comparison": results,
        "best_params": search.best_params_,
        "tuned_cv_roc_auc": round(search.best_score_, 4),
        "tuned_test_metrics": tuned_metrics,
        "tuned_confusion_matrix": tuned_cm,
    }
    with open("reports/final_results.json", "w") as f:
        json.dump(final_report, f, indent=2)

    print("\nSaved: models/churn_model.joblib (final tuned model)")
    print("Saved: reports/final_results.json, reports/model_comparison.csv")


if __name__ == "__main__":
    main()
