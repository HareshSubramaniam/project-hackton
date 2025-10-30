import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩸", layout="wide")
st.title("🩸 Diabetes Prediction using Random Forest")
st.write("Predict diabetes diagnosis using clinical and lifestyle indicators.")

df = pd.read_csv("diabetes.csv")  
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train_scaled, y_train)

st.sidebar.header("Enter Your Medical Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 50.0, 200.0, 100.0)
bp = st.sidebar.slider("Blood Pressure", 30.0, 120.0, 70.0)
skin = st.sidebar.slider("Skin Thickness", 0.0, 100.0, 20.0)
insulin = st.sidebar.slider("Insulin", 0.0, 400.0, 90.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 30)

user_input = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                          columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

user_scaled = scaler.transform(user_input)

prediction = rf_model.predict(user_scaled)[0]
prob = rf_model.predict_proba(user_scaled)[0]

st.subheader("🧬 Prediction Result")
if prediction == 1:
    st.error("⚠️ The model predicts a **high risk of Diabetes.** Please consult a doctor.")
else:
    st.success("✅ The model predicts **no diabetes risk.** Stay healthy!")

st.write("### 🔢 Prediction Confidence:")
st.write(f"Diabetes: {prob[1]*100:.2f}%")
st.write(f"No Diabetes: {prob[0]*100:.2f}%")

y_pred = rf_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")
