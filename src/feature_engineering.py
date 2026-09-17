"""
Engineer additional features from the cleaned Telco churn data.

Each feature is justified by what predictive information it adds beyond
what raw columns already carry. No leakage: every feature here is computed
only from information available at the time of prediction (no use of
future billing, no use of the target itself).
"""

import pandas as pd


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket tenure (0-72 months) into groups.
    Why: raw tenure is already numeric and useful, but churn risk is known
    to be highly non-linear early in a customer's lifecycle (very high risk
    in month 1-6, then flattening). Binning lets linear models (Logistic
    Regression) capture this non-linearity that a single numeric slope can't.
    Tree models can find this split themselves, so this mainly helps
    Logistic Regression and interpretability, not necessarily XGBoost.
    """
    df = df.copy()
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many of the 6 optional add-on services a customer has
    (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies).
    Why: a customer with many add-on services is more invested/"locked in"
    to the ecosystem than one with none - a plausible retention signal that
    isn't obvious from any single service column alone.
    """
    df = df.copy()
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    AvgMonthlySpend = TotalCharges / max(tenure, 1).
    Why: TotalCharges conflates tenure and spending rate (a long-tenure
    low-spender and a short-tenure high-spender can have the same total).
    This isolates the actual spending rate, comparable across customers
    regardless of how long they've been around.
    max(tenure, 1) avoids division by zero for the tenure==0 customers.
    """
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_tenure_group(df)
    df = add_service_count(df)
    df = add_avg_monthly_spend(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_cleaned.csv")
    df = engineer_features(df)
    df.to_csv("data/processed/telco_features.csv", index=False)
    print(f"Feature-engineered data saved: {df.shape}")
    print(df[["TenureGroup", "ServiceCount", "AvgMonthlySpend"]].head())
