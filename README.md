# 💊 AI-Powered Healthcare Cost Prediction App

This Streamlit-based web application predicts a patient’s healthcare costs based on demographic and lifestyle inputs. It uses machine learning, SHAP interpretability, and connects to Google Sheets for secure patient data storage.

---

## 🚀 Features

- 🧠 **Predict medical costs** using an XGBoost regression model
- 📝 **Input patient info**: Age, BMI, region, smoking status, etc.
- 🔍 **Explain predictions** using SHAP (feature importance)
- 📥 **Generate downloadable PDF** report per patient
- 🔐 **Log to Google Sheets** via secure Service Account
- 💡 **Health improvement tips** to reduce future costs

---

## 🧾 Tech Stack

| Tool           | Usage                                 |
|----------------|----------------------------------------|
| Streamlit      | Web app interface                     |
| XGBoost        | Predictive modeling                   |
| SHAP           | Model explainability                  |
| gspread        | Google Sheets integration             |
| FPDF           | PDF report generation                 |
| Google Cloud   | Secure service account auth           |

---

## 📊 Model Info

- **Algorithm**: XGBoost Regressor in Scikit-learn pipeline
- **Target**: Medical Charges
- **Features**:
  - Age, BMI, Children, Smoker, Sex, Region
  - Derived: Diabetes Risk Indicator
- **Performance** (approx):
  - MAE: `$4,200`
  - R² Score: `0.78`

---

## 🛠 Setup Instructions

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set up secrets (for Streamlit Cloud)

Create a `.streamlit/secrets.toml` file with:

```toml
GOOGLE_SERVICE_ACCOUNT_JSON = """
{
  "type": "service_account",
  "project_id": "your-project-id",
  ...
}
"""
```

### 3️⃣ Run locally

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 diabetes-cost-dashboard/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
└── README.md
```

---

## 🧾 Output Includes

- 🧮 Cost prediction for each patient
- 📊 SHAP visual to explain model reasoning
- 📄 Personalized PDF with cost and suggestions
- 📤 Patient data stored securely in Google Sheet

---

## 🔐 Data Privacy

- PII is securely handled and not exposed
- Data is stored in a private Google Sheet
- Only the linked service account can access the data

---

## 📌 Future Improvements

- [ ] Email the report automatically to the patient
- [ ] Add login-based access using Streamlit Authenticator
- [ ] Deploy on secure server or containerized app

---

## 👨‍⚕️ Author

**Raviteja Abbagalla** 

---

