"""
Core EDA plots and churn-rate breakdowns, saved to reports/figures/
and reports/eda_summary.md.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/processed/telco_cleaned.csv")
# Churn was mapped to 0/1 during cleaning (see data_preprocessing.py);
# use a readable label column for plotting/grouping without touching the
# numeric column other code relies on.
df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.countplot(data=df, x="ChurnLabel", ax=axes[0, 0])
axes[0, 0].set_title("Churn Distribution")

sns.histplot(data=df, x="tenure", hue="ChurnLabel", kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Tenure vs Churn")

sns.countplot(data=df, x="Contract", hue="ChurnLabel", ax=axes[0, 2])
axes[0, 2].set_title("Contract Type vs Churn")

sns.histplot(data=df, x="MonthlyCharges", hue="ChurnLabel", kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Monthly Charges vs Churn")

sns.countplot(data=df, x="InternetService", hue="ChurnLabel", ax=axes[1, 1])
axes[1, 1].set_title("Internet Service vs Churn")

sns.countplot(data=df, x="PaymentMethod", hue="ChurnLabel", ax=axes[1, 2])
axes[1, 2].tick_params(axis="x", rotation=30)
axes[1, 2].set_title("Payment Method vs Churn")

plt.tight_layout()
plt.savefig("reports/figures/eda_overview.png", dpi=120)
plt.close()

lines = ["# EDA Summary\n"]
lines.append(f"Overall churn rate: {(df['Churn']==1).mean()*100:.1f}%\n")
lines.append("## Churn rate by category\n")
for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "SeniorCitizen"]:
    rates = df.groupby(col)["Churn"].mean().mul(100).round(1)
    lines.append(f"\n**{col}**\n")
    for k, v in rates.items():
        lines.append(f"- {k}: {v}%")
    lines.append("")

with open("reports/eda_summary.md", "w") as f:
    f.write("\n".join(lines))

print("Saved reports/figures/eda_overview.png and reports/eda_summary.md")
print("\n".join(lines))
