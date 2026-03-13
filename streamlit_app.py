import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="EduPro Demand Predictor", layout="wide")

st.title("📊 EduPro Course Demand & Revenue Predictor")

st.markdown("Predict future course enrollments using machine learning.")

# Load model
try:
    model = pickle.load(open("Model/demand_model.pkl", "rb"))
except:
    st.error("Model file not found")
    st.stop()

st.sidebar.header("Course Inputs")

price = st.sidebar.slider("Course Price ($)", 0, 500, 100)
duration = st.sidebar.slider("Course Duration (hours)", 1, 50, 10)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.0)
experience = st.sidebar.slider("Instructor Experience (years)", 0, 20, 5)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.0)

if st.sidebar.button("Predict Demand"):

    features = np.array([[price, duration, rating, experience, teacher_rating]])
    prediction = model.predict(features)

    st.subheader("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Enrollments", int(prediction[0]))

    with col2:
        revenue = prediction[0] * price
        st.metric("Estimated Revenue", f"${int(revenue)}")

st.markdown("---")

st.subheader("Project Overview")

st.write("""
This dashboard predicts **course demand and revenue** for the EduPro platform using
machine learning models trained on historical course, instructor, and transaction data.
""")
