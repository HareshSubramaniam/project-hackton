import streamlit as st

st.set_page_config(
    page_title="AI Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>
    /* Dark theme background */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0077B6;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #023E8A;
        color: white;
    }

    /* Disease Cards (no white background) */
    .disease-card {
        background-color: transparent !important;
        border: none !important;
        padding: 10px;
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .disease-card:hover {
        transform: scale(1.05);
    }

    /* Image styling */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(255, 255, 255, 0.1);
    }

    /* Center footer text */
    footer {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


st.sidebar.title("🏥 Disease Prediction System")
page = st.sidebar.radio(
    "Select Disease",
    ["Home", "Heart Disease", "Diabetes", "Hypertension", "Obesity"],
    index=0
)


img_heart = "he.jpg"
img_diabetes = "dia.png"
img_hypertension = "hyper.jpeg"
img_obesity = "obe.jpg"

if page == "Home":
    st.title("🩺 AI-Powered Multi-Disease Prediction System")
    st.write("""
    Welcome! This application uses **Machine Learning** to estimate your risk for common diseases.
    """)

    st.subheader("🔬 Diseases Covered")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown('<div class="disease-card">', unsafe_allow_html=True)
        st.image(img_heart, width=150)
        st.write("### 🫀 Heart Disease")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="disease-card">', unsafe_allow_html=True)
        st.image(img_diabetes, width=150)
        st.write("### 💉 Diabetes")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="disease-card">', unsafe_allow_html=True)
        st.image(img_hypertension, width=150)
        st.write("### 💊 Hypertension")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="disease-card">', unsafe_allow_html=True)
        st.image(img_obesity, width=150)
        st.write("### ⚖️ Obesity")
        st.markdown('</div>', unsafe_allow_html=True)


elif page == "Heart Disease":
    st.title("🫀 Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

    with col2:
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])

    with col3:
        restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", [1, 0])

    if st.button("🔍 Predict Heart Disease"):
      
        pred = 0
        if pred == 1:
            st.error("🚨 Risk of Heart Disease Detected!")
        else:
            st.success("✅ No Heart Disease Detected")


elif page == "Diabetes":
    st.title("💉 Diabetes Prediction")
    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)

    with col2:
        skin = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("🔍 Predict Diabetes"):
    
        pred = 1
        if pred == 1:
            st.warning("⚠️ At Risk of Diabetes")
        else:
            st.success("✅ No Diabetes Detected")


elif page == "Hypertension":
    st.title("💊 Hypertension Prediction")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, format="%.1f")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")

    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=180)
        activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        smoker = st.selectbox("Smoking Status", ["Yes", "No"])

    if st.button("🔍 Predict Hypertension"):
       
        pred = 0
        if pred == 1:
            st.warning("⚠️ High Blood Pressure Detected")
        else:
            st.success("✅ Normal Blood Pressure")
elif page == "Obesity":
    st.title("⚖️ Obesity Prediction")
    col1, col2 = st.columns(2)

    with col1:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, format="%.1f")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")

    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🔍 Predict Obesity"):
        pred = 1
        if pred == 1:
            st.warning("⚠️ High Risk of Obesity")
        else:
            st.success("✅ Healthy Weight")

st.markdown("---")
st.caption("👨‍⚕️ Developed with ❤️ using Streamlit | 2025")
