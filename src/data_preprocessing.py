"""
Load and clean the raw Telco Customer Churn dataset.

Cleaning decisions (see reports/eda_summary.md for justification):
1. TotalCharges is stored as text with 11 blank values, all belonging to
   customers with tenure == 0 (brand-new customers who haven't been billed
   yet). Converted to numeric; the 11 blanks are imputed to 0.0, which is
   the factually correct value (no charges have accrued), not a guess.
2. customerID is dropped before modeling - it's a unique identifier with
   zero predictive value and including it would be a leakage/noise risk.
3. SeniorCitizen (0/1 int) is left as-is; it's already a clean binary flag.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided, no modifications."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning steps and return a clean copy."""
    df = df.copy()

    # Fix TotalCharges: text -> numeric, blanks -> NaN -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = df["TotalCharges"].isna()
    assert blank_mask.sum() == 11, (
        f"Expected exactly 11 blank TotalCharges rows, found {blank_mask.sum()}"
    )
    assert (df.loc[blank_mask, "tenure"] == 0).all(), (
        "Found a blank TotalCharges row where tenure != 0 - investigate before imputing"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target to binary int (kept as a separate clearly-named column)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID - unique identifier, no predictive value."""
    return df.drop(columns=["customerID"])


if __name__ == "__main__":
    raw = load_raw_data("data/raw/Telco-Customer-Churn.csv")
    cleaned = clean_data(raw)
    cleaned.to_csv("data/processed/telco_cleaned.csv", index=False)
    print(f"Cleaned data saved: {cleaned.shape}")
    print(cleaned["Churn"].value_counts(normalize=True))
