import streamlit as st
import numpy as np
import joblib

#loading model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details:")

#taking inputs
tenure = st.number_input("Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):

    #input
    features = np.array([[tenure, monthly, total]])
    features = scaler.transform(features)

    #predicting
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    #displaying result
    if prediction == 1:
        st.error(f"⚠️ High chance of churn ({probability:.2f})")
    else:
        st.success(f"✅ Customer likely to stay ({probability:.2f})")