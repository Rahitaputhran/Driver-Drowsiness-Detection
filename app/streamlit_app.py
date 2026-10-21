import streamlit as st

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title("Driver Drowsiness Detection System")
st.caption("Real-time computer vision + ML/DL portfolio project")

st.info(
    "Run `python -m app.realtime` for the live OpenCV camera application. "
    "This page is the dashboard shell for extending the project with Streamlit."
)

c1, c2, c3 = st.columns(3)
c1.metric("Architecture", "Hybrid CV + ML + DL")
c2.metric("Classifier", "XGBoost")
c3.metric("DL Backbone", "MobileNetV3Small")

st.subheader("Pipeline")
st.code(
    "Camera → Face Landmarks → EAR/MAR/PERCLOS/Head Pose → "
    "XGBoost + optional CNN → Temporal Hysteresis → Alert",
    language="text",
)

st.subheader("Key features")
st.write([
    "Eye Aspect Ratio (EAR)",
    "Mouth Aspect Ratio (MAR)",
    "PERCLOS",
    "Eye-closure duration",
    "Blink/yawn behavior",
    "Head pose",
    "Temporal smoothing and hysteresis",
])
