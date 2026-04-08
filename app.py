import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("gradient_boosting_model.joblib")
scaler = joblib.load("scaler.joblib")

# App Title
st.title("Diabetes Risk Prediction App")
st.write("Enter patient details below to assess diabetes risk")

# User Inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)

triglyceride = st.number_input("Triglyceride (mg/dL)", min_value=0.0, value=150.0)
cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=180.0)
hdl = st.number_input("HDL (mg/dL)", min_value=0.0, value=50.0)

avoid_eating_out = st.selectbox(
    "Do you avoid eating out frequently?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

sugary_food = st.selectbox(
    "Do you frequently consume sugary foods?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        'Triglyceride': [triglyceride],
        'Age': [age],
        'Cholesterol': [cholesterol],
        'HDL': [hdl],
        'Avoid_eating_out': [avoid_eating_out],
        'Sugary_food': [sugary_food]
    })

    # Fixed
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {probability:.2f}")
