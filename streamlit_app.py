import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="EduPro AI Analytics", layout="wide")

# ---------------- DARK STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.main-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4CAF50;
}

.subtitle {
    text-align:center;
    color:#AAAAAA;
    font-size:18px;
}

[data-testid="metric-container"] {
    background-color:#1c1f26;
    border:1px solid #2c2f36;
    padding:15px;
    border-radius:10px;
}

.block-container {
    padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='main-title'>EduPro Demand Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict Course Demand and Revenue using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("Model/demand_model.pkl","rb"))
except:
    st.error("Model file missing")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Course Parameters")

price = st.sidebar.slider("Course Price ($)",0,500,120)
duration = st.sidebar.slider("Course Duration (Hours)",1,40,12)
rating = st.sidebar.slider("Course Rating",1.0,5.0,4.2)
experience = st.sidebar.slider("Instructor Experience",0,20,6)
teacher_rating = st.sidebar.slider("Teacher Rating",1.0,5.0,4.3)

predict = st.sidebar.button("Predict Demand")

# ---------------- DEFAULT VALUES ----------------
enrollments = 0
revenue = 0

# ---------------- PREDICTION ----------------
if predict:

    features = np.array([[price,duration,rating,experience,teacher_rating]])
    prediction = model.predict(features)

    enrollments = int(prediction[0])
    revenue = enrollments * price

# ---------------- KPI CARDS ----------------
col1,col2,col3,col4 = st.columns(4)

col1.metric("Predicted Enrollments", enrollments)
col2.metric("Estimated Revenue", f"${revenue}")
col3.metric("Course Rating", rating)
col4.metric("Course Price", f"${price}")

st.markdown("---")

# ---------------- TABS ----------------
tab1,tab2,tab3 = st.tabs(["Demand Analysis","Revenue Forecast","Course Analytics"])

# ---------------- DEMAND ANALYSIS ----------------
with tab1:

    st.subheader("Demand vs Price")

    prices = np.arange(50,300,10)
    preds = []

    for p in prices:
        f = np.array([[p,duration,rating,experience,teacher_rating]])
        preds.append(model.predict(f)[0])

    df = pd.DataFrame({
        "Price":prices,
        "Demand":preds
    })

    fig = px.line(
        df,
        x="Price",
        y="Demand",
        title="Demand Curve",
        markers=True,
        template="plotly_dark",
        color_discrete_sequence=["#FFD700"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- REVENUE FORECAST ----------------
with tab2:

    st.subheader("Revenue Forecast")

    df["Revenue"] = df["Price"] * df["Demand"]

    fig2 = px.area(
        df,
        x="Price",
        y="Revenue",
        template="plotly_dark",
        color_discrete_sequence=["#FF4B4B"],
        title="Revenue Forecast"
    )

    st.plotly_chart(fig2,use_container_width=True)

# ---------------- COURSE ANALYTICS ----------------
with tab3:

    st.subheader("Course Category Analytics")

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
        color="Category",
        template="plotly_dark",
        color_discrete_sequence=[
            "#00E5FF",
            "#FFD700",
            "#FF4B4B",
            "#7C4DFF",
            "#00C853"
        ]
    )

    st.plotly_chart(fig3,use_container_width=True)

    st.subheader("Demand Heatmap")

    heatmap_data = np.random.rand(10,10)

    fig4 = px.imshow(
        heatmap_data,
        template="plotly_dark",
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(fig4,use_container_width=True)

