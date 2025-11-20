import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #045d04;
            font-size: 40px;
            font-weight: 700;
        }

        .predict-btn {
            display: flex;
            justify-content: center;
        }

        .signature {
            margin-top: 40px;
            text-align: center;
            font-size: 18px;
            color: #008000;
            font-style: italic;
        }

        .stButton>button {
            background-color: #05a305;
            color: white;
            padding: 10px 40px;
            font-size: 20px;
            border-radius: 10px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #008000;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown("<div class='title'>Customer Churn Prediction App by farhan </div>", unsafe_allow_html=True)

# ---------------------------
# Load trained model
# ---------------------------
model = load("xgboost_churn.joblib")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Enter Customer Information")

# Numerical inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)

# Encoded categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
geo = st.selectbox("Geography", ["France", "Spain", "Germany"])

gender_encoded = 1 if gender == "Male" else 0
geo_encoded = {"France": 0, "Spain": 1, "Germany": 2}[geo]
has_card_encoded = 1 if has_card == "Yes" else 0
is_active_encoded = 1 if is_active == "Yes" else 0

# ---------------------------
# Predict Button
# ---------------------------
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)

predict = st.button("Predict")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Model Prediction
# ---------------------------
if predict:
    input_data = [[
        credit_score, geo_encoded, gender_encoded, age, tenure,
        balance, num_products, has_card_encoded, is_active_encoded, salary
    ]]

    result = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if result == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

# ---------------------------
# Signature
# ---------------------------
st.markdown("<div class='signature'>— Mohammed Farhan</div>", unsafe_allow_html=True)
