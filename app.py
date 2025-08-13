import streamlit as st
import pandas as pd
import joblib

# تحميل الموديل
model = joblib.load("pipeline_model.pkl")

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict whether they are likely to churn")

gender_map = {"Male": 0, "Female": 1}
binary_map = {"No": 0, "Yes": 1}
internet_service_map = {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No": [0, 0, 1]}
contract_map = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
payment_map = {
    "Bank transfer (automatic)": [1, 0, 0, 0],
    "Credit card (automatic)": [0, 1, 0, 0],
    "Electronic check": [0, 0, 1, 0],
    "Mailed check": [0, 0, 0, 1]
}

with st.form("churn_form"):
    gender = st.selectbox("Gender", list(gender_map.keys()))
    SeniorCitizen = st.selectbox("Senior Citizen", list(binary_map.keys()))
    Partner = st.selectbox("Partner", list(binary_map.keys()))
    Dependents = st.selectbox("Dependents", list(binary_map.keys()))
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", list(binary_map.keys()))
    MultipleLines = st.selectbox("Multiple Lines", list(binary_map.keys()))
    OnlineSecurity = st.selectbox("Online Security", list(binary_map.keys()))
    OnlineBackup = st.selectbox("Online Backup", list(binary_map.keys()))
    DeviceProtection = st.selectbox("Device Protection", list(binary_map.keys()))
    TechSupport = st.selectbox("Tech Support", list(binary_map.keys()))
    StreamingTV = st.selectbox("Streaming TV", list(binary_map.keys()))
    StreamingMovies = st.selectbox("Streaming Movies", list(binary_map.keys()))
    PaperlessBilling = st.selectbox("Paperless Billing", list(binary_map.keys()))
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=75.35)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=903.25)

    InternetService = st.selectbox("Internet Service", list(internet_service_map.keys()))
    Contract = st.selectbox("Contract", list(contract_map.keys()))
    PaymentMethod = st.selectbox("Payment Method", list(payment_map.keys()))

    submit_button = st.form_submit_button("Predict")

if submit_button:

    new_data = pd.DataFrame([{
        'gender': gender_map[gender],
        'SeniorCitizen': binary_map[SeniorCitizen],
        'Partner': binary_map[Partner],
        'Dependents': binary_map[Dependents],
        'tenure': tenure,
        'PhoneService': binary_map[PhoneService],
        'MultipleLines': binary_map[MultipleLines],
        'OnlineSecurity': binary_map[OnlineSecurity],
        'OnlineBackup': binary_map[OnlineBackup],
        'DeviceProtection': binary_map[DeviceProtection],
        'TechSupport': binary_map[TechSupport],
        'StreamingTV': binary_map[StreamingTV],
        'StreamingMovies': binary_map[StreamingMovies],
        'PaperlessBilling': binary_map[PaperlessBilling],
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'InternetService_DSL': internet_service_map[InternetService][0],
        'InternetService_Fiber optic': internet_service_map[InternetService][1],
        'InternetService_No': internet_service_map[InternetService][2],
        'Contract_Month-to-month': contract_map[Contract][0],
        'Contract_One year': contract_map[Contract][1],
        'Contract_Two year': contract_map[Contract][2],
        'PaymentMethod_Bank transfer (automatic)': payment_map[PaymentMethod][0],
        'PaymentMethod_Credit card (automatic)': payment_map[PaymentMethod][1],
        'PaymentMethod_Electronic check': payment_map[PaymentMethod][2],
        'PaymentMethod_Mailed check': payment_map[PaymentMethod][3]
    }])

    prediction = model.predict(new_data)[0]
    if prediction == 1:
        st.error("🚨 The customer is likely to churn!")
    else:
        st.success("✅ The customer is likely to stay.")
