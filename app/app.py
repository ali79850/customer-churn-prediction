"""
Streamlit app for individual customer churn prediction.
Run with: streamlit run app/app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.05)
total_charges = st.number_input(
    "Total Charges ($)", min_value=0.0, value=float(monthly_charges * max(tenure, 1)), step=0.05
)

if st.button("Predict Churn Risk", type="primary"):
    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    raw_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    df = pd.DataFrame([customer])[raw_cols]
    df = engineer_features(df)

    proba = model.predict_proba(df)[0, 1]
    risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.35 else "Low"
    prediction = "Customer is predicted to be at elevated risk of churn." \
        if proba >= 0.5 else "Customer is predicted to stay."

    st.divider()
    st.metric("Churn Probability", f"{proba * 100:.1f}%")
    st.metric("Risk Level", risk)
    st.write(prediction)

    st.subheader("Top factors typically associated with this risk profile")
    factors = []
    if contract == "Month-to-month":
        factors.append("+ Month-to-month contract (largest single risk driver in this model)")
    else:
        factors.append("- Longer-term contract (reduces risk)")
    if internet_service == "Fiber optic":
        factors.append("+ Fiber optic internet service")
    if online_security == "No":
        factors.append("+ No online security add-on")
    if tech_support == "No":
        factors.append("+ No tech support add-on")
    if tenure < 12:
        factors.append("+ Short tenure (<12 months)")
    elif tenure > 48:
        factors.append("- Long tenure (>48 months, reduces risk)")

    for f in factors:
        st.write(f)
    st.caption(
        "These are model-driven associations learned from historical data, "
        "not proof that any single factor causes churn."
    )
