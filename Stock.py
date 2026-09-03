import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="SIH26216 Live Stock Prediction Dashboard", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AI-Powered Real-Time Stock Prediction & Analysis")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("Market Controls")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()
time_period = st.sidebar.selectbox("Select Historical Range", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)

st.sidebar.header("ML Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)

# Fetch Live Data using yfinance
@st.cache_data
def load_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

data_load_state = st.text(f"Fetching live data for {ticker_symbol}...")
df = load_data(ticker_symbol, time_period)
data_load_state.text("")

if df.empty:
    st.error(f"Invalid ticker symbol '{ticker_symbol}' or no market data found. Try standard codes like AAPL, TSLA, RELIANCE.NS, or BTC-USD.")
else:
    # Key Metrics Display
    latest_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2]
    price_change = latest_price - prev_price
    pct_change = (price_change / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"${latest_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
    col2.metric("Day High", f"${df['High'].iloc[-1]:.2f}")
    col3.metric("Day Low", f"${df['Low'].iloc[-1]:.2f}")
    col4.metric("Total Volume", f"{int(df['Volume'].iloc[-1]):,}")

    st.markdown("### 📊 Interactive Candlestick Chart & Technical Indicators")
    
    # Calculate Technical Indicators (SMA & Bollinger Bands)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)

    # Plotly Advanced Interactive Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Market Candlestick'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1.5), name='20 SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Upper Bollinger'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Lower Bollinger', fill='tonexty'))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        title=f"{ticker_symbol} Technical Analysis & Price Action"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Machine Learning Section
    st.markdown("### 🤖 Random Forest Price Prediction Model")
    st.write(f"Model status: Ready to train using live fetched data for {ticker_symbol}.")
    
    if st.button("Run Prediction Model"):
        # Simple feature engineering for ML
        model_df = df[['Close']].copy()
        model_df['Prediction'] = model_df['Close'].shift(-1)
        model_df.dropna(inplace=True)
        
        X = np.array(model_df[['Close']][:-1])
        y = np.array(model_df['Prediction'][:-1])
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        st.success("Random Forest Model trained successfully on live market data!")
        
        # Display chart of predictions vs actuals
        predictions = model.predict(X_test)
        pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}, index=model_df.index[split_idx+1:])
        st.line_chart(pred_df)

    # Raw Data view expander
    with st.expander("View Raw Market Data Table"):
        st.dataframe(df.tail(20))