import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

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

with st.form("input_form"):
    name = st.text_input("Patient Name", value="John Doe")
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
    st.subheader(f"💰 Predicted Medical Cost for {name}: **${prediction:,.2f}**")

    st.markdown("---")
    st.subheader("📌 Why this prediction? (SHAP)")

    processed_input = model.named_steps['preprocessor'].transform(input_df)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_input, columns=feature_names)
    individual_shap = explainer(processed_df)

    fig_individual, ax_individual = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(individual_shap[0], show=False)
    st.pyplot(fig_individual)

    # ------------------------------
    # Suggestions
    # ------------------------------
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
    if len(suggestions) == 0:
        suggestions.append("• Maintain your current healthy lifestyle!")

    for s in suggestions:
        st.write(s)

    # ------------------------------
    # PDF Report Generator
    # ------------------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_title("Medical Cost Prediction Report")
    pdf.cell(200, 10, txt=f"Patient Report: {name}", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Predicted Medical Cost: ${prediction:,.2f}", ln=True, align="L")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Input Summary:", ln=True)
    for k, v in input_df.iloc[0].items():
        pdf.cell(200, 10, txt=f"{k.capitalize()}: {v}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Recommendations:", ln=True)
    for s in suggestions:
        pdf.multi_cell(0, 10, txt=s)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.download_button(
        label="📥 Download Patient Report (PDF)",
        data=pdf_output,
        file_name=f"{name.replace(' ', '_')}_report.pdf",
        mime="application/pdf"
    )
