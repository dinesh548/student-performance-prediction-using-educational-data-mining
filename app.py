# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Data Simulation (fallback)
# -----------------------------
@st.cache_data
def generate_data(n_samples=200):
    np.random.seed(42)
    attendance = np.random.uniform(50, 100, n_samples)
    study_hours = np.random.uniform(0, 40, n_samples)
    prev_grade = np.random.uniform(40, 100, n_samples)

    final_grade = (
        0.3 * attendance +
        1.2 * study_hours +
        0.5 * prev_grade +
        np.random.normal(0, 5, n_samples)
    )
    final_grade = np.clip(final_grade, 0, 100)

    df = pd.DataFrame({
        "Attendance_Rate": attendance,
        "Study_Hours_Per_Week": study_hours,
        "Previous_Grade": prev_grade,
        "Final_Grade": final_grade
    })
    return df

# -----------------------------
# Model Training
# -----------------------------
@st.cache_data
def train_model(df, model_type="RandomForest"):
    X = df.drop("Final_Grade", axis=1)
    y = df["Final_Grade"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mae, r2

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Student Performance Prediction Dashboard", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    # Standardize column names
    df.rename(columns={
        "Study_Hours_per_Week": "Study_Hours_Per_Week",
        "Past_Exam_Scores": "Previous_Grade",
        "Final_Exam_Score": "Final_Grade"
    }, inplace=True)

    # Keep only numeric columns for training
    df = df[["Attendance_Rate", "Study_Hours_Per_Week", "Previous_Grade", "Final_Grade"]]
else:
    df = generate_data()
    st.info("Using synthetic dataset since no file was uploaded.")

# Sidebar Inputs (aligned with dataset)
st.sidebar.header("Enter Student Data")
attendance_input = st.sidebar.slider("Attendance Rate (%)", 50, 100, 75)
study_hours_input = st.sidebar.slider("Study Hours per Week", 0, 40, 20)
prev_grade_input = st.sidebar.slider("Previous Grade", 40, 100, 70)

model_choice = st.sidebar.selectbox("Select Model", ["RandomForest", "LinearRegression"])

# Train model
model, scaler, mae, r2 = train_model(df, model_choice)

# Model Info
st.sidebar.subheader("📊 Model Info")
st.sidebar.write(f"**Mean Absolute Error:** {mae:.2f}")
st.sidebar.write(f"**R-squared:** {r2:.2f}")

# Prediction
user_data = pd.DataFrame({
    "Attendance_Rate": [attendance_input],
    "Study_Hours_Per_Week": [study_hours_input],
    "Previous_Grade": [prev_grade_input]
})

if model_choice == "LinearRegression":
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
else:
    prediction = model.predict(user_data)[0]

prediction = np.clip(prediction, 0, 100)

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Study Hours vs Final Grade")
    fig = px.scatter(df, x="Study_Hours_Per_Week", y="Final_Grade",
                     color="Attendance_Rate",
                     title="Correlation between Study Hours and Final Grade")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    st.metric(label="Predicted Final Grade", value=f"{prediction:.2f}")
    st.metric(label="Class Average", value=f"{df['Final_Grade'].mean():.2f}")

# Feature Importance (only for RandomForest)
if model_choice == "RandomForest":
    st.subheader("🔍 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": df.drop("Final_Grade", axis=1).columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

# Predict Button
if st.sidebar.button("Predict"):
    st.success(f"✅ Predicted Final Grade: {prediction:.2f}")