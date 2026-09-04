"""
PredictStock — Institutional-Grade Stock Analysis & Prediction Dashboard
Built with Streamlit, yfinance, Plotly, and scikit-learn.
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictStock | AI Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS — FINTECH DARK THEME
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: linear-gradient(180deg, #0b0e14 0%, #0e1117 100%);
        color: #e6e9ef;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}

    /* ---------- Header ---------- */
    .ps-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 26px;
        background: linear-gradient(90deg, #121722 0%, #161b26 100%);
        border: 1px solid #232a37;
        border-radius: 14px;
        margin-bottom: 22px;
    }
    .ps-header h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        color: #f3f5f9;
        letter-spacing: 0.3px;
    }
    .ps-header p {
        margin: 2px 0 0 0;
        font-size: 13px;
        color: #8b93a7;
    }
    .ps-badge {
        background: #1b2a20;
        color: #4ade80;
        border: 1px solid #234a30;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
    }

    /* ---------- Metric Cards ---------- */
    div[data-testid="stMetric"] {
        background: #131722;
        border: 1px solid #232a37;
        padding: 16px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #8b93a7 !important; font-size: 13px !important; }
    div[data-testid="stMetricValue"] { color: #f3f5f9 !important; font-weight: 700 !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0d1017;
        border-right: 1px solid #1f2530;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f3f5f9;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 18px;
    }

    /* ---------- Section headers ---------- */
    .ps-section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f3f5f9;
        margin: 26px 0 6px 0;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
    }
    .ps-sub {
        color: #8b93a7;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ---------- Footer ---------- */
    .ps-footer {
        margin-top: 40px;
        padding: 16px 0;
        border-top: 1px solid #232a37;
        color: #6b7280;
        font-size: 12px;
        text-align: center;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stButton > button:hover { background: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
now_str = datetime.datetime.now().strftime("%b %d, %Y — %H:%M")
st.markdown(
    f"""
    <div class="ps-header">
        <div>
            <h1>📈 PredictStock</h1>
            <p>AI-Powered Real-Time Market Analysis &amp; Short-Term Prediction Dashboard</p>
        </div>
        <div class="ps-badge">● Live Data · {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — TICKER PRESETS
# ----------------------------------------------------------------------------
st.sidebar.header("Market Controls")

PRESETS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "NVDA", "GOOGL"]
preset_choice = st.sidebar.selectbox(
    "Quick Ticker Presets", ["Custom (type below)"] + PRESETS
)

default_ticker = "AAPL" if preset_choice == "Custom (type below)" else preset_choice
ticker_symbol = st.sidebar.text_input(
    "Stock Ticker Symbol", value=default_ticker
).upper().strip()

# ----------------------------------------------------------------------------
# SIDEBAR — TIMEFRAME SELECTION (Colab-style logic)
# ----------------------------------------------------------------------------
st.sidebar.header("Timeframe & Date Controls")
timeframe_option = st.sidebar.selectbox(
    "Select Historical Mode",
    [
        "Option 1: Current Year Data",
        "Option 2: Specific Selected Year",
        "Option 3: Custom Date Range / Multi-Year",
    ],
)

current_year = datetime.datetime.now().year

if timeframe_option == "Option 1: Current Year Data":
    start_date = f"{current_year}-01-01"
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")

elif timeframe_option == "Option 2: Specific Selected Year":
    selected_year = st.sidebar.number_input(
        "Select Year", min_value=2000, max_value=current_year, value=current_year - 1
    )
    start_date = f"{selected_year}-01-01"
    end_date = f"{selected_year}-12-31"

else:
    start_year = st.sidebar.number_input(
        "Start Year", min_value=2000, max_value=current_year, value=current_year - 2
    )
    start_date = st.sidebar.date_input(
        "Start Date", datetime.date(int(start_year), 1, 1)
    )
    end_date = st.sidebar.date_input("End Date", datetime.datetime.today().date())

# ----------------------------------------------------------------------------
# SIDEBAR — ML MODEL PARAMETERS
# ----------------------------------------------------------------------------
st.sidebar.header("ML Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 300, 100)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 30, 10)

with st.sidebar.expander("ℹ️ How does the prediction model work?"):
    st.write(
        "This dashboard trains a **Random Forest Regressor** — an ensemble of "
        "decision trees — using each day's closing price to predict the *next* "
        "day's closing price. It is a short-term, pattern-based estimate, **not** "
        "financial advice. Accuracy depends heavily on market volatility and the "
        "selected date range."
    )

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


with st.spinner(f"Fetching live market data for {ticker_symbol}..."):
    df = load_data(ticker_symbol, start_date, end_date)

if df.empty:
    st.error(
        f"⚠️ No market data found for '{ticker_symbol}' in the selected date range. "
        "Check the ticker symbol or choose a different timeframe."
    )
    st.stop()

# ----------------------------------------------------------------------------
# KEY METRICS
# ----------------------------------------------------------------------------
latest_price = float(df["Close"].iloc[-1])
prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100 if prev_price else 0.0

st.markdown('<div class="ps-section-title">Market Snapshot</div>', unsafe_allow_html=True)
st.markdown(f'<div class="ps-sub">{ticker_symbol} · {start_date} → {end_date}</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Latest Close",
    f"${latest_price:,.2f}",
    f"{price_change:+.2f} ({pct_change:+.2f}%)",
)
col2.metric("Day High", f"${float(df['High'].iloc[-1]):,.2f}")
col3.metric("Day Low", f"${float(df['Low'].iloc[-1]):,.2f}")
col4.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
col5.metric("Period Range", f"${float(df['Low'].min()):,.2f} – ${float(df['High'].max()):,.2f}")

# ----------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------------------------------
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["Bollinger_Mid"] = df["Close"].rolling(window=20).mean()
df["Bollinger_Upper"] = df["Bollinger_Mid"] + (df["Close"].rolling(window=20).std() * 2)
df["Bollinger_Lower"] = df["Bollinger_Mid"] - (df["Close"].rolling(window=20).std() * 2)

# ----------------------------------------------------------------------------
# CANDLESTICK + VOLUME CHART
# ----------------------------------------------------------------------------
st.markdown('<div class="ps-section-title">📊 Price Action & Technical Indicators</div>', unsafe_allow_html=True)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.03,
    subplot_titles=(f"{ticker_symbol} — Candlestick with Moving Averages", "Volume"),
)

fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
    ),
    row=1, col=1,
)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color="#facc15", width=1.4), name="SMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color="#38bdf8", width=1.4), name="SMA 200"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], line=dict(color="#c084fc", width=1.2, dash="dot"), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Upper"], line=dict(color="#6b7280", width=1, dash="dash"), name="Bollinger Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Lower"], line=dict(color="#6b7280", width=1, dash="dash"), name="Bollinger Lower", fill="tonexty", fillcolor="rgba(107,114,128,0.08)"), row=1, col=1)

volume_colors = np.where(df["Close"] >= df["Open"], "#22c55e", "#ef4444")
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=volume_colors), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    height=650,
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#e6e9ef"),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_xaxes(showgrid=True, gridcolor="#1f2530", row=1, col=1)
fig.update_xaxes(showgrid=True, gridcolor="#1f2530", row=2, col=1)
fig.update_yaxes(showgrid=True, gridcolor="#1f2530", row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor="#1f2530", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# MACHINE LEARNING — RANDOM FOREST PREDICTION
# ----------------------------------------------------------------------------
st.markdown('<div class="ps-section-title">🤖 Random Forest Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps-sub">Predicts next-day closing price from historical closes. '
    'Trained fresh on the data currently loaded above.</div>',
    unsafe_allow_html=True,
)

run_model = st.button("▶ Run Prediction Model", use_container_width=False)

if run_model:
    model_df = df[["Close"]].copy()
    model_df["Prediction"] = model_df["Close"].shift(-1)
    model_df.dropna(inplace=True)

    if len(model_df) > 10:
        X = np.array(model_df[["Close"]][:-1])
        y = np.array(model_df["Prediction"][:-1])

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        with st.spinner("Training Random Forest model..."):
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.success("Model trained successfully on live market data.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error", f"${mae:,.2f}")
        m2.metric("R² Score", f"{r2:.3f}")
        m3.metric("Test Samples", f"{len(y_test)}")

        pred_dates = model_df.index[split_idx + 1:]
        pred_df = pd.DataFrame(
            {"Actual": y_test, "Predicted": predictions}, index=pred_dates
        )

        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=pred_df.index, y=pred_df["Actual"],
            line=dict(color="#38bdf8", width=2), name="Actual Price",
        ))
        pred_fig.add_trace(go.Scatter(
            x=pred_df.index, y=pred_df["Predicted"],
            line=dict(color="#facc15", width=2, dash="dot"), name="Predicted Price",
        ))
        pred_fig.update_layout(
            template="plotly_dark",
            height=420,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#e6e9ef"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title="Actual vs. Predicted Closing Price (Test Set)",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        pred_fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
        pred_fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
        st.plotly_chart(pred_fig, use_container_width=True)

        csv_data = pred_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
        st.download_button(
            label="⬇ Download Prediction Results (CSV)",
            data=csv_data,
            file_name=f"{ticker_symbol}_predictions.csv",
            mime="text/csv",
        )
    else:
        st.warning("Not enough data points in this range to train the ML model. Try a wider date range.")

# ----------------------------------------------------------------------------
# RAW DATA + FULL EXPORT
# ----------------------------------------------------------------------------
with st.expander("📄 View Raw Market Data Table"):
    st.dataframe(df.tail(50), use_container_width=True)
    full_csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇ Download Full Dataset (CSV)",
        data=full_csv,
        file_name=f"{ticker_symbol}_raw_data.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="ps-footer">
        PredictStock Dashboard · Data via Yahoo Finance (yfinance) · For educational purposes only — not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)