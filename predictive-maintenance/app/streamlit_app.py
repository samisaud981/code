"""Interactive Streamlit UI for predictive maintenance predictions."""
from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("⚙️ Predictive Maintenance Dashboard")
st.caption("Interactive UI for live failure risk scoring")

api_url = st.sidebar.text_input("API URL", "http://api:8000/predict")

col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider("Temperature", 30.0, 140.0, 72.0)
    vibration = st.slider("Vibration", 0.0, 5.0, 1.0)
with col2:
    pressure = st.slider("Pressure", 10.0, 60.0, 32.0)
    humidity = st.slider("Humidity", 5.0, 95.0, 45.0)
with col3:
    rpm = st.slider("RPM", 400.0, 2200.0, 1400.0)
    degradation = st.slider("Degradation", 0.0, 1.0, 0.5)

payload = {
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "humidity": humidity,
    "rpm": rpm,
    "degradation": degradation,
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Output")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Failure Probability", f"{result['failure_probability']:.3f}")
        k2.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
        k3.metric("Binary Prediction", str(result["binary_prediction"]))
        k4.metric("Drift Detected", str(result["drift_detected"]))

        st.json({"input": payload, "output": result})
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
