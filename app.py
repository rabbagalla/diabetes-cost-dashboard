import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -------------------------
# Google Sheets Setup
# -------------------------
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gcred.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("Patient_Cost_Data").sheet1
    return sheet
    # ------------------------------
    # Google Sheets Integration
    # ------------------------------
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    import json
    import os

    # Load credentials from secrets
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    # Open the sheet (must already exist and be shared with the service account)
    sheet = client.open("Patient_Cost_Records").sheet1

    # Prepare data to store
    sheet.append_row([
        name, age, height_cm, weight_kg, bmi,
        children, smoker, sex, region,
        diabetes_risk, f"${prediction:,.2f}", phone, address
    ])

# -------------------------
# Streamlit Page Setup
# -------------------------
st.set_page_config(page_title="Healthcare Cost Predictor", layout="centered")
st.title("💊 AI-Powered Healthcare Cost Prediction")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)

    def diabetes_risk(row):
        if row['bmi'] > 30 and row['age'] > 45:
            return 1
        elif row['bmi'] > 35:
            return 1
        elif row['smoker'] == 'yes' and row['bmi'] > 28:
            return 1
        else:
            return 0

    df['diabetes_risk'] = df.apply(diabetes_risk, axis=1)
    return df

df = load_data()

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'diabetes_risk']
target = 'charges'
X = df[features]
y = df[target]

cat_cols = ['sex', 'smoker', 'region']
num_cols = ['age', 'bmi', 'children', 'diabetes_risk']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("MAE", f"${mae:,.2f}")
st.sidebar.metric("R² Score", f"{r2:.2f}")

explainer = shap.Explainer(model.named_steps['xgb'], model.named_steps['preprocessor'].transform(X_train))
shap_values = explainer(model.named_steps['preprocessor'].transform(X_train))

fig_global, ax_global = plt.subplots(figsize=(8, 4))
shap.plots.bar(shap_values, show=False)
st.sidebar.subheader("🔍 Top Cost Drivers")
st.sidebar.pyplot(fig_global)

# -------------------------
# User Input Form
# -------------------------
with st.form("input_form"):
    name = st.text_input("Patient Name", value="John Doe")
    address = st.text_input("Patient Address")
    phone = st.text_input("Phone Number (10 digits)")

    age = st.slider("Age", 18, 100, 40)
    height_cm = st.number_input("Height (in cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight_kg = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    st.markdown(f"**Calculated BMI:** {bmi}")

    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    diabetes_risk = 1 if (bmi > 30 and age > 45) or (smoker == 'yes' and bmi > 28) else 0

    submitted = st.form_submit_button("Predict Cost")

# -------------------------
# Prediction + Save + PDF
# -------------------------
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
    st.subheader(f"💰 Predicted Medical Cost for {name}: **${prediction:,.2f}**")

    # Save to Google Sheet
    try:
        sheet = connect_sheet()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([
            name, phone, address, age, height_cm, weight_kg, bmi, smoker, sex, region,
            children, diabetes_risk, round(prediction, 2), timestamp
        ])
        st.success("✅ Patient data stored in Google Sheet.")
    except Exception as e:
        st.error(f"❌ Error saving to Google Sheet: {e}")

    # SHAP plot
    st.markdown("---")
    st.subheader("📌 Why this prediction? (SHAP)")
    processed_input = model.named_steps['preprocessor'].transform(input_df)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_input, columns=feature_names)
    individual_shap = explainer(processed_df)

    fig_individual, ax_individual = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(individual_shap[0], show=False)
    st.pyplot(fig_individual)

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Suggestions to Reduce Future Costs")
    suggestions = []
    if smoker == "yes":
        suggestions.append("• Stop smoking to reduce long-term risks.")
    if bmi > 30:
        suggestions.append("• Consider a nutrition and exercise program to manage BMI.")
    if diabetes_risk == 1:
        suggestions.append("• Enroll in a diabetes prevention or care management plan.")
    if age > 60:
        suggestions.append("• Schedule regular screenings and wellness checkups.")
    if not suggestions:
        suggestions.append("• Maintain your current healthy lifestyle!")

    for s in suggestions:
        st.write(s)

    # PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Medical Cost Prediction Report")
    pdf.cell(200, 10, txt=f"Patient Report: {name}", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Predicted Medical Cost: ${prediction:,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Phone: {phone}", ln=True)
    pdf.cell(200, 10, txt=f"Address: {address}", ln=True)
    pdf.ln(5)
    for k, v in input_df.iloc[0].items():
        line = f"{k.capitalize()}: {v}"
        safe_line = line.encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(200, 10, txt=safe_line, ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Suggestions to Reduce Future Costs:", ln=True)
    for s in suggestions:
        safe_s = s.replace("•", "-").encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, txt=safe_s)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output = BytesIO(pdf_bytes)

    st.download_button(
        label="📥 Download Patient Report (PDF)",
        data=pdf_output,
        file_name=f"{name.replace(' ', '_')}_report.pdf",
        mime="application/pdf"
    )
