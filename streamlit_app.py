import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="EduPro Analytics", layout="wide")

# Header
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
📊 EduPro Course Demand & Revenue Predictor
</h1>
<p style='text-align: center;'>
AI-powered insights for course planning
</p>
""", unsafe_allow_html=True)

# Load model
try:
    model = pickle.load(open("Model/demand_model.pkl","rb"))
except:
    st.error("Model file not found")
    st.stop()

# Sidebar inputs
st.sidebar.header("🎛 Course Configuration")

price = st.sidebar.slider("Course Price ($)",0,500,120)
duration = st.sidebar.slider("Course Duration (Hours)",1,40,10)
rating = st.sidebar.slider("Course Rating",1.0,5.0,4.2)
experience = st.sidebar.slider("Instructor Experience (Years)",0,20,6)
teacher_rating = st.sidebar.slider("Teacher Rating",1.0,5.0,4.5)

predict = st.sidebar.button("🚀 Predict Demand")

# Prediction
if predict:

    features = np.array([[price,duration,rating,experience,teacher_rating]])
    prediction = model.predict(features)

    enrollments = int(prediction[0])
    revenue = enrollments * price

    st.subheader("📈 Prediction Results")

    col1,col2,col3 = st.columns(3)

    col1.metric("👨‍🎓 Predicted Enrollments", enrollments)
    col2.metric("💰 Estimated Revenue", f"${revenue}")
    col3.metric("⭐ Course Rating", rating)

# Visualization section
st.markdown("---")
st.subheader("📊 Demand Trend Simulation")

prices = np.arange(50,300,20)
preds = []

for p in prices:
    f = np.array([[p,duration,rating,experience,teacher_rating]])
    preds.append(model.predict(f)[0])

chart_data = pd.DataFrame({
    "Price": prices,
    "Predicted Enrollments": preds
})

st.line_chart(chart_data.set_index("Price"))

st.markdown("---")

st.subheader("📚 Project Overview")

st.write("""
This dashboard predicts **course demand and revenue** for the EduPro learning platform.

The model analyzes multiple factors:
- Course price
- Course duration
- Course rating
- Instructor experience
- Teacher rating

Using machine learning, EduPro can **forecast enrollment demand and optimize course pricing strategies.**
""")
