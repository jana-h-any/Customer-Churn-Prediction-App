# app.py
# This Streamlit app provides a user interface for a customer churn prediction model,
# using a pre-trained Decision Tree model and manual feature encoding.

import streamlit as st
import pandas as pd
import joblib

# --- Load the trained model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model using joblib."""
    try:
        # Load the model from the file
        model = joblib.load("pipeline_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Error: The model file 'desisionTree_gridsearch.pkl' was not found. "
                 "Please ensure it is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# Exit if the model didn't load correctly
if model is None:
    st.stop()


# --- Streamlit UI ---

st.title("Customer Churn Prediction App")
st.markdown("""
    Enter customer details below to predict whether they are likely to churn,
    using a model trained with a Decision Tree and Grid Search.
""")
st.divider()

# --- Input Fields Grouped for Better UI ---
st.header("Customer Profile")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="0 for No, 1 for Yes")
    Partner = st.selectbox("Has a Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()
st.header("Service and Billing Information")

col3, col4 = st.columns(2)
with col3:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ])

with col4:
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, step=0.01)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0, step=0.01)

# --- Feature Engineering to Match Model Input ---
# This section manually creates the features exactly as the model expects them.
# The `gender` column was likely converted to `gender_encoded` in the notebook.
gender_encoded = 1 if gender == "Female" else 0
Partner_encoded = 1 if Partner == "Yes" else 0
Dependents_encoded = 1 if Dependents == "Yes" else 0
PhoneService_encoded = 1 if PhoneService == "Yes" else 0
PaperlessBilling_encoded = 1 if PaperlessBilling == "Yes" else 0

# One-hot encoding for MultipleLines
MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0
MultipleLines_No = 1 if MultipleLines == "No" else 0
MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0

# One-hot encoding for InternetService
InternetService_DSL = 1 if InternetService == "DSL" else 0
InternetService_Fiber = 1 if InternetService == "Fiber optic" else 0
InternetService_No = 1 if InternetService == "No" else 0

# One-hot encoding for OnlineSecurity
OnlineSecurity_Yes = 1 if OnlineSecurity == "Yes" else 0
OnlineSecurity_No = 1 if OnlineSecurity == "No" else 0
OnlineSecurity_No_internet_service = 1 if OnlineSecurity == "No internet service" else 0

# One-hot encoding for OnlineBackup
OnlineBackup_Yes = 1 if OnlineBackup == "Yes" else 0
OnlineBackup_No = 1 if OnlineBackup == "No" else 0
OnlineBackup_No_internet_service = 1 if OnlineBackup == "No internet service" else 0

# One-hot encoding for DeviceProtection
DeviceProtection_Yes = 1 if DeviceProtection == "Yes" else 0
DeviceProtection_No = 1 if DeviceProtection == "No" else 0
DeviceProtection_No_internet_service = 1 if DeviceProtection == "No internet service" else 0

# One-hot encoding for TechSupport
TechSupport_Yes = 1 if TechSupport == "Yes" else 0
TechSupport_No = 1 if TechSupport == "No" else 0
TechSupport_No_internet_service = 1 if TechSupport == "No internet service" else 0

# One-hot encoding for StreamingTV
StreamingTV_Yes = 1 if StreamingTV == "Yes" else 0
StreamingTV_No = 1 if StreamingTV == "No" else 0
StreamingTV_No_internet_service = 1 if StreamingTV == "No internet service" else 0

# One-hot encoding for StreamingMovies
StreamingMovies_Yes = 1 if StreamingMovies == "Yes" else 0
StreamingMovies_No = 1 if StreamingMovies == "No" else 0
StreamingMovies_No_internet_service = 1 if StreamingMovies == "No internet service" else 0


# One-hot encoding for Contract
Contract_Month = 1 if Contract == "Month-to-month" else 0
Contract_One = 1 if Contract == "One year" else 0
Contract_Two = 1 if Contract == "Two year" else 0

# One-hot encoding for PaymentMethod
PM_Bank = 1 if PaymentMethod == "Bank transfer (automatic)" else 0
PM_Credit = 1 if PaymentMethod == "Credit card (automatic)" else 0
PM_Electronic = 1 if PaymentMethod == "Electronic check" else 0
PM_Mailed = 1 if PaymentMethod == "Mailed check" else 0

# --- Build input data DataFrame with all encoded features ---
# The column names here must precisely match the features used for training.
input_data = pd.DataFrame([{
    'SeniorCitizen': SeniorCitizen,
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender_Female': gender_encoded, # Assuming this is how 'gender' was one-hot encoded
    'Partner_Yes': Partner_encoded,
    'Dependents_Yes': Dependents_encoded,
    'PhoneService_Yes': PhoneService_encoded,
    'MultipleLines_No': MultipleLines_No,
    'MultipleLines_No phone service': MultipleLines_No_phone_service,
    'MultipleLines_Yes': MultipleLines_Yes,
    'InternetService_DSL': InternetService_DSL,
    'InternetService_Fiber optic': InternetService_Fiber,
    'InternetService_No': InternetService_No,
    'OnlineSecurity_No': OnlineSecurity_No,
    'OnlineSecurity_No internet service': OnlineSecurity_No_internet_service,
    'OnlineSecurity_Yes': OnlineSecurity_Yes,
    'OnlineBackup_No': OnlineBackup_No,
    'OnlineBackup_No internet service': OnlineBackup_No_internet_service,
    'OnlineBackup_Yes': OnlineBackup_Yes,
    'DeviceProtection_No': DeviceProtection_No,
    'DeviceProtection_No internet service': DeviceProtection_No_internet_service,
    'DeviceProtection_Yes': DeviceProtection_Yes,
    'TechSupport_No': TechSupport_No,
    'TechSupport_No internet service': TechSupport_No_internet_service,
    'TechSupport_Yes': TechSupport_Yes,
    'StreamingTV_No': StreamingTV_No,
    'StreamingTV_No internet service': StreamingTV_No_internet_service,
    'StreamingTV_Yes': StreamingTV_Yes,
    'StreamingMovies_No': StreamingMovies_No,
    'StreamingMovies_No internet service': StreamingMovies_No_internet_service,
    'StreamingMovies_Yes': StreamingMovies_Yes,
    'Contract_Month-to-month': Contract_Month,
    'Contract_One year': Contract_One,
    'Contract_Two year': Contract_Two,
    'PaperlessBilling_Yes': PaperlessBilling_encoded,
    'PaymentMethod_Bank transfer (automatic)': PM_Bank,
    'PaymentMethod_Credit card (automatic)': PM_Credit,
    'PaymentMethod_Electronic check': PM_Electronic,
    'PaymentMethod_Mailed check': PM_Mailed,
}])


# --- Prediction button and result display ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. (Probability: {prediction_proba[1]:.2f})")
        else:
            st.success(f"✅ The customer is likely to stay. (Probability: {prediction_proba[0]:.2f})")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
