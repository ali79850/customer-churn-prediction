"""
SHAP explainability: global feature influence + local individual explanations.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

model = joblib.load("models/churn_model.joblib")
X_test = pd.read_csv("data/processed/X_test.csv")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_df)

# Global explanation: summary plot (bar - mean |SHAP value| per feature)
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig("reports/figures/shap_global_importance.png", dpi=120)
plt.close()

# Global explanation: beeswarm (shows direction of effect, not just magnitude)
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig("reports/figures/shap_beeswarm.png", dpi=120)
plt.close()

print("Saved global SHAP plots to reports/figures/")

# Local explanation: one specific high-risk customer
proba = model.predict_proba(X_test)[:, 1]
high_risk_idx = proba.argmax()
print(f"\nHighest predicted churn risk: row {high_risk_idx}, probability = {proba[high_risk_idx]:.4f}")

plt.figure()
shap.force_plot(
    explainer.expected_value, shap_values[high_risk_idx],
    X_test_df.iloc[high_risk_idx], matplotlib=True, show=False,
)
plt.tight_layout()
plt.savefig("reports/figures/shap_local_high_risk_customer.png", dpi=120)
plt.close()
print("Saved local explanation plot for highest-risk customer.")
