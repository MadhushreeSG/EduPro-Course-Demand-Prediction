import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="EduPro AI Dashboard", layout="wide")

# Header
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#6C63FF;
}
.subtitle {
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>EduPro Course Demand & Revenue Forecasting</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict future course enrollments and revenue using machine learning</div>", unsafe_allow_html=True)

st.markdown("---")

# Load model
try:
    model = pickle.load(open("Model/demand_model.pkl","rb"))
except:
    st.error("Model file not found")
    st.stop()

# Sidebar
st.sidebar.header("Course Parameters")

price = st.sidebar.slider("Course Price ($)", 0, 500, 100)
duration = st.sidebar.slider("Course Duration (Hours)", 1, 40, 10)
rating = st.sidebar.slider("Course Rating", 1.0, 5.0, 4.0)
experience = st.sidebar.slider("Instructor Experience (Years)", 0, 20, 5)
teacher_rating = st.sidebar.slider("Teacher Rating", 1.0, 5.0, 4.2)

predict = st.sidebar.button("Predict Demand")

# Prediction
if predict:

    features = np.array([[price, duration, rating, experience, teacher_rating]])
    prediction = model.predict(features)

    enrollments = int(prediction[0])
    revenue = enrollments * price

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    col1.metric("Predicted Enrollments", enrollments)
    col2.metric("Estimated Revenue ($)", revenue)

# Demand vs Price Chart
st.markdown("---")
st.subheader("Demand vs Price Analysis")

prices = np.arange(50,300,20)
predictions = []

for p in prices:
    f = np.array([[p, duration, rating, experience, teacher_rating]])
    predictions.append(model.predict(f)[0])

chart_data = pd.DataFrame({
    "Price": prices,
    "Predicted Enrollments": predictions
})

st.line_chart(chart_data.set_index("Price"))

# Feature importance section
st.markdown("---")
st.subheader("Key Demand Drivers")

importance_data = pd.DataFrame({
    "Feature":[
        "Course Price",
        "Course Duration",
        "Course Rating",
        "Instructor Experience",
        "Teacher Rating"
    ],
    "Importance":[30,15,25,20,10]
})

st.bar_chart(importance_data.set_index("Feature"))

