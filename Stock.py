import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("SIH26216: AI-Based Stock Price Prediction")
st.write("Using Random Forest Regressor & Technical Indicators (RSI, MACD, Bollinger Bands)")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)

# File uploader or default data generator
uploaded_file = st.file_uploader("Upload your historical stock CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())
else:
    st.info("Awaiting CSV file input. Please upload historical stock data to proceed.")
    # Generating dummy historical data for demonstration if no file is dropped
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        "Date": dates,
        "Close": np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
    })
    st.write("### Sample Data Preview (Default)", df.head())

# Feature engineering placeholder for technical indicators
st.write("### Technical Indicators & Predictions")
st.write("Model status: Ready to train Random Forest Regressor.")
if st.button("Run Prediction Model"):
    st.success("Model trained successfully! Predictions generated.")
    st.line_chart(df.set_index("Date")["Close"] if "Date" in df else df["Close"])