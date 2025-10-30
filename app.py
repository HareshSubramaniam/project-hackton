import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Hypertension ML Predictor", page_icon="💓", layout="wide")
st.title("💓 Hypertension Risk Prediction App")
st.write("This app predicts the risk of hypertension using multiple ML models.")

df = pd.read_csv("hypertension.csv")

X = df[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

st.sidebar.header("🧍 Input Health Details")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

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
                         columns=['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                                  'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])

user_scaled = scaler.transform(user_data)
selected_model = models[model_choice]
prediction = selected_model.predict(user_scaled)
probability = selected_model.predict_proba(user_scaled)[0][1] if hasattr(selected_model, "predict_proba") else None

st.subheader("🩺 Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Hypertension ({model_choice})")
else:
    st.success(f"✅ Low Risk of Hypertension ({model_choice})")

if probability is not None:
    st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

st.subheader("📊 Model Accuracy Comparison")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
st.dataframe(acc_df.style.highlight_max(axis=0, color="lightgreen"))

st.bar_chart(acc_df.set_index("Model"))
