import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import joblib

# Load model
model = joblib.load("xgb_pipeline.pkl")

st.set_page_config(page_title="Healthcare Cost Predictor", layout="centered")
st.title("💊 AI-Powered Healthcare Cost Prediction")
st.markdown("Estimate future medical costs based on patient characteristics. Useful for insurers & value-based care planning.")

# Input form
with st.form("input_form"):
    age = st.slider("Age", 18, 100, 40)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    diabetes_risk = 1 if (bmi > 30 and age > 45) or (smoker == 'yes' and bmi > 28) else 0

    submitted = st.form_submit_button("Predict Cost")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "sex": sex,
        "region": region,
        "diabetes_risk": diabetes_risk
    }])

    prediction = model.predict(input_df)[0]
    st.subheader(f"💰 Predicted Medical Cost: **${prediction:,.2f}**")
