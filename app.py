import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load your trained model
loaded_model = joblib.load("gradient_boosting_model.joblib")

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', loaded_model)
])

# Save pipeline
joblib.dump(pipeline, "pipeline_model.joblib")
# Load the pipeline (scaler + model)
pipeline = joblib.load("pipeline_model.joblib")

st.title("Diabetes Risk Prediction App")
st.write("Enter patient details below to assess diabetes risk")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
triglyceride = st.number_input("Triglyceride (mg/dL)", min_value=0.0, value=150.0)
cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=180.0)
hdl = st.number_input("HDL (mg/dL)", min_value=0.0, value=50.0)
avoid_eating_out = st.selectbox("Do you avoid eating out frequently?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
sugary_food = st.selectbox("Do you frequently consume sugary foods?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

if st.button("Predict"):
    input_data = pd.DataFrame([[
        triglyceride, age, cholesterol, hdl, avoid_eating_out, sugary_food
    ]], columns=[
        'Triglyceride','Age','Cholesterol','HDL','Avoid_eating_out','Sugary_food'
    ])

    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {probability:.2f}")
