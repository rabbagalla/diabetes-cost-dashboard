import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ------------------------------
# PAGE SETUP
# ------------------------------
st.set_page_config(page_title="Healthcare Cost Predictor", layout="centered")
st.title("💊 AI-Powered Healthcare Cost Prediction")
st.markdown("Estimate medical costs based on patient details. Useful for insurers & population health planning.")

# ------------------------------
# LOAD AND PREPARE DATA
# ------------------------------
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

# ------------------------------
# MODEL TRAINING
# ------------------------------
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

# ------------------------------
# METRICS DISPLAY (SIDEBAR)
# ------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("MAE", f"${mae:,.2f}")
st.sidebar.metric("R² Score", f"{r2:.2f}")

# ------------------------------
# SHAP GLOBAL FEATURE IMPORTANCE
# ------------------------------
explainer = shap.Explainer(model.named_steps['xgb'], model.named_steps['preprocessor'].transform(X_train))
shap_values = explainer(model.named_steps['preprocessor'].transform(X_train))

fig_global, ax_global = plt.subplots(figsize=(8, 4))
shap.plots.bar(shap_values, show=False)
st.sidebar.subheader("🔍 Top Cost Drivers")
st.sidebar.pyplot(fig_global)

# ------------------------------
# USER INPUT FORM
# ------------------------------
with st.form("input_form"):
    age = st.slider("Age", 18, 100, 40)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker?", ["yes", "no"])
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    diabetes_risk = 1 if (bmi > 30 and age > 45) or (smoker == 'yes' and bmi > 28) else 0

    submitted = st.form_submit_button("Predict Cost")

# ------------------------------
# COST PREDICTION
# ------------------------------
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

    # SHAP EXPLANATION FOR INDIVIDUAL
    st.markdown("---")
    st.subheader("📌 Why this prediction? (SHAP)")

    processed_input = model.named_steps['preprocessor'].transform(input_df)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_input, columns=feature_names)

    # ✅ Fix: Use global explainer instead of new one
    individual_shap = explainer(processed_df)

    fig_individual, ax_individual = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(individual_shap[0], show=False)
    st.pyplot(fig_individual)
