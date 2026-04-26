import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('attrition_model88.pkl', 'rb'))

# load scaler and columns
scaler = pickle.load(open('scaler88.pkl', 'rb'))
columns = pickle.load(open('columns88.pkl', 'rb'))

# title
st.title("HR Attrition Prediction App")

# input variables
age = st.number_input("Age", min_value=18, max_value=60, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 2)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.selectbox("OverTime", ("Yes", "No"))

# encode
OverTime_Yes = 1 if overtime == "Yes" else 0

# create dataframe
input_features = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "JobSatisfaction": [job_satisfaction],
    "WorkLifeBalance": [work_life_balance],
    "YearsAtCompany": [years_at_company],
    "OverTime_Yes": [OverTime_Yes]
})

# ensure all expected columns exist
for col in columns:
    if col not in input_features:
        input_features[col] = 0

# reorder columns
input_features = input_features[columns]

# scale input
input_scaled = scaler.transform(input_features)

# predictions
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")
