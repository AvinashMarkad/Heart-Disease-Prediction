import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
model = joblib.load('logistic_Regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart ❤️ Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user inputs.")

# User Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["YES", "NO"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction logic
if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in raw_input:
            raw_input[col] = 0

    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts a high likelihood of heart disease.")
    else:
        st.success("The model predicts a low likelihood of heart disease.")
