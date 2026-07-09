import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="House Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Rent Prediction")
st.subheader("Machine Learning & Neural Network")

st.write("""
Welcome to the House Rent Prediction project.

This application predicts house rent using:
- 🤖 Random Forest Regressor
- 🧠 Artificial Neural Network
""")

st.success("Project setup completed successfully! ✅")