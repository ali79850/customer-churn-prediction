"""
Prediction function for individual customers, using the exact same
feature engineering + preprocessing pipeline used during training.
"""

import os

import joblib
import pandas as pd

from feature_engineering import engineer_features

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "..", "models", "churn_model.joblib")

RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def predict_churn(customer: dict, model=None) -> dict:
    """
    customer: dict with the raw customer fields (see RAW_COLUMNS).
    Returns: {"churn_probability": float, "prediction": str, "risk_level": str}
    """
    if model is None:
        model = joblib.load(MODEL_PATH)

    missing = set(RAW_COLUMNS) - set(customer.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = pd.DataFrame([customer])[RAW_COLUMNS]
    df = engineer_features(df)

    proba = model.predict_proba(df)[0, 1]

    if proba >= 0.6:
        risk = "High"
    elif proba >= 0.35:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": "Likely to churn" if proba >= 0.5 else "Likely to stay",
        "risk_level": risk,
    }


if __name__ == "__main__":
    example_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.70,
        "TotalCharges": 151.65,
    }
    result = predict_churn(example_customer)
    print(f"Churn Probability: {result['churn_probability'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Prediction: {result['prediction']}")
