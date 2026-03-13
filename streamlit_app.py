import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("EduPro Course Demand Predictor")

try:
    model = pickle.load(open("Model/demand_model.pkl","rb"))
    st.success("Model loaded successfully")
except:
    st.error("Model file not found")
    st.stop()

price = st.slider("Course Price",0,500,100)
duration = st.slider("Course Duration",1,50,10)
rating = st.slider("Course Rating",1.0,5.0,4.0)
experience = st.slider("Instructor Experience",0,20,5)
teacher_rating = st.slider("Teacher Rating",1.0,5.0,4.0)

if st.button("Predict Enrollment"):

    features = np.array([[price,duration,rating,experience,teacher_rating]])
    prediction = model.predict(features)

    st.success(f"Predicted Enrollment: {int(prediction[0])}")
