import numpy as np
import streamlit as st
from model import run_fcm, W_trained

st.set_page_config(layout="centered")

# DEBUG LINE (very important)
st.write("APP RELOADED")

st.title("🎓 Student Academic Performance Prediction (FCM + NHL)")
st.markdown("Adjust the factors below (0–1 scale):")

# -------------------------------
# Sliders (must match dataset)
# -------------------------------
study = st.slider("Study_Effort", 0.0, 1.0, 0.5)
attendance = st.slider("Attendance", 0.0, 1.0, 0.5)
prior = st.slider("Prior_Ability", 0.0, 1.0, 0.5)
motivation = st.slider("Motivation", 0.0, 1.0, 0.5)
stress = st.slider("Stress", 0.0, 1.0, 0.5)
sleep = st.slider("Sleep_Quality", 0.0, 1.0, 0.5)
peer = st.slider("Peer_Support", 0.0, 1.0, 0.5)
time = st.slider("Time_Management", 0.0, 1.0, 0.5)
digital = st.slider("Digital_Distraction", 0.0, 1.0, 0.5)
teaching = st.slider("Teaching_Effectiveness", 0.0, 1.0, 0.5)

# Debug values
st.write("Current Inputs:")
st.write([study, attendance, prior, motivation, stress, sleep, peer, time, digital, teaching])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Academic Performance"):

    state = np.array([
        study,
        attendance,
        prior,
        motivation,
        stress,
        sleep,
        peer,
        time,
        digital,
        teaching,
        0.0   # placeholder (ignored by model)
    ])

    # ✅ Direct prediction (model already applies sigmoid)
    performance = run_fcm(state, W_trained)

    st.success(f"Predicted Academic Performance Score: {performance:.3f}")
     # Progress bar
    st.progress(float(performance))

    # -------------------------------
    # Risk Interpretation
    # -------------------------------
    if performance <= 0.45:
        st.error("🔴 High Risk: Student performance may be poor")

    elif performance <= 0.65:
        st.warning("🟡 Medium Risk: Student needs improvement")

    else:
        st.success("🟢 Good Performance: Student likely to perform well")