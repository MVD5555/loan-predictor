import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load('loan_predictor_model.pkl')

st.title("Loan Approval Predictor")
st.write("Enter the applicant details below to predict loan approval status.")

st.subheader("Applicant Details")
no_of_dependents = st.slider("Number of Dependents", min_value=0, max_value=5, value=2, step=1)
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self-Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (INR)", min_value=0, value=5000000, step=100000)
loan_amount = st.number_input("Loan Amount (INR)", min_value=0, value=10000000, step=100000)
loan_term = st.slider("Loan Term (Years)", min_value=2, max_value=20, value=10, step=1)
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=600, step=1)

st.subheader("Asset Values (INR)")
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=5000000, step=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=2000000, step=100000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=10000000, step=100000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=3000000, step=100000)

input_data = pd.DataFrame({
    'no_of_dependents': [no_of_dependents],
    'education': [0 if education == "Graduate" else 1],
    'self_employed': [1 if self_employed == "Yes" else 0],
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value]
})

if st.button("Predict Loan Status"):
    pred = rf_model.predict(input_data)
    result = "Approved" if pred[0] == 0 else "Rejected"
    st.success(f"Loan Status: **{result}**")

if st.checkbox("Show Input Data"):
    st.write("Input Data for Prediction:")
    st.dataframe(input_data)


st.markdown("---")
st.write("Created by Mayur Vijay Dumbre")  