import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("loan_model.pkl")
encoders = joblib.load("loan_encoders.pkl")
target_encoder = joblib.load("loan_target_encoder.pkl")

st.title("🏦 Loan Approval Prediction")

st.write("Fill in the applicant details and check if the loan will be approved ✅")

# --- Input fields ---
gender = st.selectbox("Gender", encoders["Gender"].classes_)
married = st.selectbox("Married", encoders["Married"].classes_)
dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)
education = st.selectbox("Education", encoders["Education"].classes_)
self_employed = st.selectbox("Self Employed", encoders["Self_Employed"].classes_)

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term (Months)", [120, 180, 240, 300, 360])
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", encoders["Property_Area"].classes_)

# --- Prepare input ---
if st.button("Predict Loan Status"):
    input_data = pd.DataFrame({
        "Gender": [encoders["Gender"].transform([gender])[0]],
        "Married": [encoders["Married"].transform([married])[0]],
        "Dependents": [encoders["Dependents"].transform([dependents])[0]],
        "Education": [encoders["Education"].transform([education])[0]],
        "Self_Employed": [encoders["Self_Employed"].transform([self_employed])[0]],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [encoders["Property_Area"].transform([property_area])[0]]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    if result == "Y":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
