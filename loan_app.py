# ------------------------------
# Streamlit Loan Approval Predictor (with Encoders)
# ------------------------------

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('loan_model.pkl')
encoders = joblib.load('loan_encoders.pkl')
target_encoder = joblib.load('loan_target_encoder.pkl')

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("Predict loan approval for single applicants or upload a CSV for batch predictions.")

# ---------------- Single Applicant Prediction ----------------
st.subheader("Single Applicant Prediction")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", 1000, 50000, 5000)
    coapplicant_income = st.number_input("Coapplicant Income", 0, 30000, 2000)
    loan_amount = st.number_input("Loan Amount (in 1000s)", 50, 700, 150)
    loan_term = st.selectbox("Loan Amount Term (months)", [120, 180, 240, 360])
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Prepare input DataFrame
new_applicant = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Encode categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols:
    new_applicant[col] = encoders[col].transform(new_applicant[col])

# Predict single applicant
if st.button("Predict Single Applicant"):
    prediction = model.predict(new_applicant)[0]
    probability = model.predict_proba(new_applicant)[0][1]

    st.subheader("Prediction Result")
    st.write("Predicted Status:", target_encoder.inverse_transform([prediction])[0])
    st.write("Approval Probability: {:.2f}%".format(probability*100))

# ---------------- Batch Prediction ----------------
st.subheader("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV for multiple applicants", type=["csv"])

if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)

    # Encode categorical columns
    for col in categorical_cols:
        if col in df_batch.columns:
            df_batch[col] = encoders[col].transform(df_batch[col])

    # Make predictions
    preds = model.predict(df_batch)
    probs = model.predict_proba(df_batch)[:,1]

    df_batch['Predicted_Status'] = target_encoder.inverse_transform(preds)
    df_batch['Approval_Probability'] = probs

    st.success("✅ Batch Prediction Completed")
    st.dataframe(df_batch)

    # Download predictions
    csv = df_batch.to_csv(index=False).encode()
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

