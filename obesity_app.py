import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Obesity Prediction", page_icon="🍔", layout="wide")
st.title("🍔 Obesity Category Prediction (XGBoost Model)")
st.write("Predict obesity category based on lifestyle and body metrics.")

df = pd.read_csv("obesity.csv")

X = df.drop(columns=["ObesityCategory"])
y = df["ObesityCategory"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
gender = 1 if gender == "Male" else 0
height = st.sidebar.slider("Height (cm)", 120, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 160, 70)

bmi = weight / ((height / 100) ** 2)
st.sidebar.write(f"**Calculated BMI:** {bmi:.2f}")

physical_activity = st.sidebar.slider("Physical Activity Level (1 = Low, 5 = High)", 1, 5, 3)

user_input = pd.DataFrame([[age, gender, height, weight, bmi, physical_activity]],
                          columns=["Age", "Gender", "Height", "Weight", "BMI", "PhysicalActivityLevel"])

user_scaled = scaler.transform(user_input)

prediction = xgb_model.predict(user_scaled)[0]
probs = xgb_model.predict_proba(user_scaled)[0]

st.subheader("🏥 Prediction Result")

category_map = {
    0: "Normal Weight",
    1: "Overweight",
    2: "Obese"
}

st.write(f"### 🧩 Predicted Category: **{category_map.get(prediction, 'Unknown')}**")

st.write("### 🔢 Prediction Confidence:")
for i, label in category_map.items():
    st.progress(float(probs[i]))
    st.write(f"{label}: {probs[i]*100:.2f}%")

y_pred = xgb_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")
