import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction (Random Forest Model)")
st.write("Predict **heart disease risk** based on biometric and lifestyle data.")

df = pd.read_csv(r"C:\Users\dhara\HypertensionML\cleaned_heart_disease.csv")

X = df[['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
        'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption',
        'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
        'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']]
y = df['Heart Disease Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

st.sidebar.header("Enter Your Health Details")

Age = st.sidebar.slider("Age", 18, 100, 40)
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Gender = 1 if Gender == "Male" else 0
Blood_Pressure = st.sidebar.slider("Blood Pressure", 90, 200, 130)
Cholesterol_Level = st.sidebar.slider("Cholesterol Level", 100, 400, 200)
Exercise_Habits = st.sidebar.selectbox("Exercise Regularly?", ("Yes", "No"))
Exercise_Habits = 1 if Exercise_Habits == "Yes" else 0
Smoking = st.sidebar.selectbox("Smoking?", ("Yes", "No"))
Smoking = 1 if Smoking == "Yes" else 0
Family_Heart_Disease = st.sidebar.selectbox("Family Heart Disease?", ("Yes", "No"))
Family_Heart_Disease = 1 if Family_Heart_Disease == "Yes" else 0
Diabetes = st.sidebar.selectbox("Diabetes?", ("Yes", "No"))
Diabetes = 1 if Diabetes == "Yes" else 0
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
High_BP = st.sidebar.selectbox("High Blood Pressure?", ("Yes", "No"))
High_BP = 1 if High_BP == "Yes" else 0
Low_HDL = st.sidebar.selectbox("Low HDL Cholesterol?", ("Yes", "No"))
Low_HDL = 1 if Low_HDL == "Yes" else 0
High_LDL = st.sidebar.selectbox("High LDL Cholesterol?", ("Yes", "No"))
High_LDL = 1 if High_LDL == "Yes" else 0
Alcohol = st.sidebar.slider("Alcohol Consumption (0–3)", 0, 3, 1)
Stress = st.sidebar.slider("Stress Level (0–3)", 0, 3, 1)
Sleep = st.sidebar.slider("Sleep Hours", 3, 10, 7)
Sugar = st.sidebar.slider("Sugar Consumption (0–3)", 0, 3, 1)
Triglyceride = st.sidebar.slider("Triglyceride Level", 50, 500, 150)
FBS = st.sidebar.slider("Fasting Blood Sugar", 60, 200, 100)
CRP = st.sidebar.slider("CRP Level", 0.0, 20.0, 5.0)
Homo = st.sidebar.slider("Homocysteine Level", 0.0, 30.0, 10.0)

user_data = pd.DataFrame([[Age, Gender, Blood_Pressure, Cholesterol_Level, Exercise_Habits,
                           Smoking, Family_Heart_Disease, Diabetes, BMI, High_BP, Low_HDL,
                           High_LDL, Alcohol, Stress, Sleep, Sugar, Triglyceride, FBS,
                           CRP, Homo]],
                         columns=X.columns)

user_scaled = scaler.transform(user_data)
prediction = rf_model.predict(user_scaled)
probability = rf_model.predict_proba(user_scaled)[0][1]

st.subheader("🫀 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")
