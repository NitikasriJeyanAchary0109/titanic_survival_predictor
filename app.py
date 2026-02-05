import streamlit as st
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = joblib.load("model.pkl")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
}
.card {
    background: #020617;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(56,189,248,0.2);
}
.title {
    text-align: center;
    font-size: 40px;
    color: #38bdf8;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}
.label {
    color: #e5e7eb;
    font-weight: 600;
}
.result-success {
    color: #22c55e;
    font-size: 26px;
    text-align: center;
    font-weight: bold;
}
.result-fail {
    color: #ef4444;
    font-size: 26px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ UI ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter passenger details to predict survival</div>', unsafe_allow_html=True)

# ------------------ INPUTS ------------------
pclass = st.selectbox("🎟 Passenger Class", [1, 2, 3])

sex = st.selectbox("👤 Sex", ["male", "female"])
sex_encoded = 1 if sex == "male" else 0

age = st.number_input("🎂 Age", min_value=0.0, max_value=100.0, value=25.0)

sibsp = st.number_input("👨‍👩‍👧 Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)

parch = st.number_input("👶 Parents / Children Aboard", min_value=0, max_value=10, value=0)

fare = st.number_input("💰 Fare", min_value=0.0, max_value=600.0, value=30.0)

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown('<p class="result-success">✅ Passenger Survived</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result-fail">❌ Passenger Did Not Survive</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
