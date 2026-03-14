import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# PAGE CONFIG
st.set_page_config(page_title="EduPro AI Analytics", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>

.block-container{
padding-top:1rem;
}

.main-title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#4CAF50;
}

.subtitle{
text-align:center;
color:#9aa0a6;
font-size:18px;
margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='main-title'>EduPro Demand Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict Course Demand and Revenue using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- LOAD MODEL ----------
try:
    model = pickle.load(open("Model/demand_model.pkl","rb"))
except:
    st.error("Model file missing")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Course Parameters")

price = st.sidebar.slider("Course Price ($)",0,500,120)
duration = st.sidebar.slider("Course Duration (Hours)",1,40,12)
rating = st.sidebar.slider("Course Rating",1.0,5.0,4.2)
experience = st.sidebar.slider("Instructor Experience",0,20,6)
teacher_rating = st.sidebar.slider("Teacher Rating",1.0,5.0,4.3)

predict = st.sidebar.button("Predict Demand")

# ---------- PREDICTION ----------
enrollments = 0
revenue = 0

if predict:

    features = np.array([[price,duration,rating,experience,teacher_rating]])
    prediction = model.predict(features)

    enrollments = int(prediction[0])
    revenue = enrollments * price

# ---------- KPI CARDS ----------
col1,col2,col3,col4 = st.columns(4)

col1.metric("Predicted Enrollments", enrollments)
col2.metric("Estimated Revenue", f"${revenue}")
col3.metric("Course Rating", rating)
col4.metric("Course Price", f"${price}")

st.markdown("---")

# ---------- DEMAND DATA ----------
prices = np.arange(50,300,10)

preds = []

for p in prices:
    f = np.array([[p,duration,rating,experience,teacher_rating]])
    preds.append(model.predict(f)[0])

df = pd.DataFrame({
    "Price":prices,
    "Demand":preds
})

df["Revenue"] = df["Price"] * df["Demand"]

# ---------- SIDE BY SIDE GRAPHS ----------
col1, col2 = st.columns(2)

with col1:

    fig1 = px.line(
        df,
        x="Price",
        y="Demand",
        title="Demand vs Price"
    )

    fig1.update_traces(
        line_color="yellow",
        line_width=4
    )

    fig1.update_layout(height=320)

    st.plotly_chart(fig1, use_container_width=True)


with col2:

    fig2 = px.area(
        df,
        x="Price",
        y="Revenue",
        title="Revenue Forecast"
    )

    fig2.update_traces(
        line_color="red",
        fillcolor="rgba(255,0,0,0.5)"
    )

    fig2.update_layout(height=320)

    st.plotly_chart(fig2, use_container_width=True)


# ---------- CATEGORY ANALYTICS ----------
st.markdown("### Course Category Analytics")

categories = ["Programming","Business","Data Science","Design","Marketing"]

category_demand = np.random.randint(100,500,5)

cat_df = pd.DataFrame({
    "Category":categories,
    "Enrollments":category_demand
})

fig3 = px.bar(
    cat_df,
    x="Category",
    y="Enrollments",
    color="Category"
)

fig3.update_layout(height=350)

st.plotly_chart(fig3,use_container_width=True)


# ---------- HEATMAP ----------
st.markdown("### Demand Heatmap")

heatmap_data = np.random.rand(10,10)

fig4 = px.imshow(
    heatmap_data,
    color_continuous_scale="Viridis"
)

fig4.update_layout(height=350)

st.plotly_chart(fig4,use_container_width=True)

