"""
Threshold analysis and feature importance for the final tuned model.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

model = joblib.load("models/churn_model.joblib")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

y_proba = model.predict_proba(X_test)[:, 1]

print("=== THRESHOLD ANALYSIS ===")
print(f"{'Threshold':<10}{'Precision':<11}{'Recall':<9}{'F1':<9}{'N Flagged':<10}")
threshold_results = []
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    n_flagged = y_pred_t.sum()
    print(f"{t:<10}{p:<11.4f}{r:<9.4f}{f1:<9.4f}{n_flagged:<10}")
    threshold_results.append({"threshold": t, "precision": p, "recall": r,
                               "f1": f1, "n_flagged": int(n_flagged)})

pd.DataFrame(threshold_results).to_csv("reports/threshold_analysis.csv", index=False)

# Feature importance (from the XGBoost classifier inside the pipeline)
print("\n=== FEATURE IMPORTANCE (XGBoost gain) ===")
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
print(imp_df.head(15).to_string(index=False))
imp_df.to_csv("reports/feature_importance.csv", index=False)
