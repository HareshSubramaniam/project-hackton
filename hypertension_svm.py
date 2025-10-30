import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Hypertension Prediction", page_icon="💉", layout="wide")
st.title("💉 Hypertension Risk Prediction (SVM Model)")
st.write("Predict whether a person is at **risk of hypertension** based on health parameters.")

df = pd.read_csv("hypertension.csv")

X = df[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes',
         'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

st.sidebar.header("Enter Your Health Details")

male = st.sidebar.selectbox("Gender", ("Male", "Female"))
male = 1 if male == "Male" else 0
age = st.sidebar.slider("Age", 18, 100, 30)
currentSmoker = st.sidebar.selectbox("Are you a smoker?", ("No", "Yes"))
currentSmoker = 1 if currentSmoker == "Yes" else 0
cigsPerDay = st.sidebar.slider("Cigarettes per day", 0, 60, 0)
BPMeds = st.sidebar.selectbox("On BP Medication?", ("No", "Yes"))
BPMeds = 1 if BPMeds == "Yes" else 0
diabetes = st.sidebar.selectbox("Diabetes?", ("No", "Yes"))
diabetes = 1 if diabetes == "Yes" else 0
totChol = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
sysBP = st.sidebar.slider("Systolic BP", 90, 200, 120)
diaBP = st.sidebar.slider("Diastolic BP", 50, 120, 80)
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
heartRate = st.sidebar.slider("Heart Rate", 50, 150, 80)
glucose = st.sidebar.slider("Glucose Level", 50, 250, 100)

user_data = pd.DataFrame([[male, age, currentSmoker, cigsPerDay, BPMeds,
                           diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                         columns=X.columns)

user_scaled = scaler.transform(user_data)
prediction = svm_model.predict(user_scaled)
probability = svm_model.predict_proba(user_scaled)[0][1]

st.subheader("🩺 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ High Risk of Hypertension")
else:
    st.success("✅ Low Risk of Hypertension")

st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")
