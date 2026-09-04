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
    .stApp {
        background: linear-gradient(180deg, #0b0e14 0%, #0e1117 100%);
        color: #e6e9ef;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}

    .ps-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 18px 26px;
        background: linear-gradient(90deg, #121722 0%, #161b26 100%);
        border: 1px solid #232a37; border-radius: 14px; margin-bottom: 22px;
    }
    .ps-header h1 { font-size: 26px; font-weight: 700; margin: 0; color: #f3f5f9; letter-spacing: 0.3px; }
    .ps-header p { margin: 2px 0 0 0; font-size: 13px; color: #8b93a7; }
    .ps-badge {
        background: #1b2a20; color: #4ade80; border: 1px solid #234a30;
        padding: 6px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #131722; border: 1px solid #232a37; padding: 16px 18px;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #8b93a7 !important; font-size: 13px !important; }
    div[data-testid="stMetricValue"] { color: #f3f5f9 !important; font-weight: 700 !important; }

    section[data-testid="stSidebar"] { background: #0d1017; border-right: 1px solid #1f2530; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f3f5f9; font-size: 14px; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 18px;
    }

    .ps-section-title {
        font-size: 18px; font-weight: 700; color: #f3f5f9;
        margin: 26px 0 6px 0; border-left: 4px solid #3b82f6; padding-left: 10px;
    }
    .ps-sub { color: #8b93a7; font-size: 13px; margin-bottom: 14px; }

    .ps-signal {
        display: inline-block; padding: 10px 20px; border-radius: 10px;
        font-weight: 700; font-size: 16px; letter-spacing: 0.4px;
    }
    .ps-buy { background: #16241a; color: #4ade80; border: 1px solid #245a34; }
    .ps-sell { background: #2a1416; color: #f87171; border: 1px solid #5c2226; }
    .ps-hold { background: #241f14; color: #facc15; border: 1px solid #5c4f22; }

    .ps-footer {
        margin-top: 40px; padding: 16px 0; border-top: 1px solid #232a37;
        color: #6b7280; font-size: 12px; text-align: center;
    }

    .stButton > button {
        background: #3b82f6; color: white; border: none; border-radius: 8px;
        padding: 8px 18px; font-weight: 600;
    }
    .stButton > button:hover { background: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CURRENCY SUPPORT
# ----------------------------------------------------------------------------
CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "USDT": "₮",
}


@st.cache_data(show_spinner=False, ttl=3600)
def get_native_currency(ticker: str) -> str:
    """Best-effort detection of the currency a ticker is quoted in."""
    try:
        info = yf.Ticker(ticker).fast_info
        cur = getattr(info, "currency", None)
        if not cur and isinstance(info, dict):
            cur = info.get("currency")
        if cur:
            return cur.upper()
    except Exception:
        pass
    return "USD"


@st.cache_data(show_spinner=False, ttl=1800)
def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """Fetch a live conversion rate. USDT is treated as a USD-pegged stablecoin."""
    a = "USD" if from_currency == "USDT" else from_currency
    b = "USD" if to_currency == "USDT" else to_currency
    if a == b:
        return 1.0
    try:
        pair = yf.download(f"{a}{b}=X", period="5d", progress=False)
        if not pair.empty:
            return float(pair["Close"].dropna().iloc[-1])
        inverse = yf.download(f"{b}{a}=X", period="5d", progress=False)
        if not inverse.empty:
            return 1.0 / float(inverse["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 1.0  # graceful fallback — no conversion rather than a crash


def fmt(value: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}"


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
now_str = datetime.datetime.now().strftime("%b %d, %Y — %H:%M")
st.markdown(
    f"""
    <div class="ps-header">
        <div>
            <h1>📈 PredictStock</h1>
            <p>AI-Powered Market Analysis, Multi-Currency Pricing &amp; Short-Term Prediction</p>
        </div>
        <div class="ps-badge">● Live Data · {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — TICKER + CURRENCY
# ----------------------------------------------------------------------------
st.sidebar.header("Market Controls")

PRESETS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "NVDA", "GOOGL"]
preset_choice = st.sidebar.selectbox("Quick Ticker Presets", ["Custom (type below)"] + PRESETS)
default_ticker = "AAPL" if preset_choice == "Custom (type below)" else preset_choice
ticker_symbol = st.sidebar.text_input("Stock Ticker Symbol", value=default_ticker).upper().strip()

native_currency = get_native_currency(ticker_symbol)

CURRENCY_OPTIONS = ["USD", "INR", "EUR", "GBP", "JPY", "USDT"]
display_currency = st.sidebar.selectbox(
    "Display Currency",
    CURRENCY_OPTIONS,
    index=CURRENCY_OPTIONS.index(native_currency) if native_currency in CURRENCY_OPTIONS else 0,
    help="Prices are quoted natively in the exchange's currency, then converted "
         "live using current FX rates. USDT is shown at parity with USD.",
)

fx_rate = get_fx_rate(native_currency, display_currency)
st.sidebar.caption(
    f"Native currency: **{native_currency}** → Displaying in **{display_currency}** "
    f"(rate: 1 {native_currency} = {fx_rate:.4f} {display_currency})"
)

# ----------------------------------------------------------------------------
# SIDEBAR — TIMEFRAME SELECTION
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
    start_date = st.sidebar.date_input("Start Date", datetime.date(int(start_year), 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.datetime.today().date())

if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR — ML MODEL PARAMETERS
# ----------------------------------------------------------------------------
st.sidebar.header("ML Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 300, 100)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 30, 10)

with st.sidebar.expander("ℹ️ How does the prediction model work?"):
    st.write(
        "A **Random Forest Regressor** learns from each day's Open, High, Low, "
        "Close, Volume, and moving averages to estimate the *next* day's closing "
        "price. Data is split **chronologically** (no shuffling) so the model is "
        "never trained on the future. This is a short-term statistical estimate, "
        "**not** financial advice."
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


try:
    with st.spinner(f"Fetching live market data for {ticker_symbol}..."):
        df = load_data(ticker_symbol, start_date, end_date)
except Exception as e:
    st.error(f"⚠️ Could not fetch data for '{ticker_symbol}'. Details: {e}")
    st.stop()

if df.empty or "Close" not in df.columns:
    st.error(
        f"⚠️ No market data found for '{ticker_symbol}' in the selected date range. "
        "Check the ticker symbol or choose a different timeframe."
    )
    st.stop()

if len(df) < 30:
    st.warning(
        "⚠️ Very little historical data in this range — indicators and the "
        "prediction model will be less reliable. Consider widening the date range."
    )

# ----------------------------------------------------------------------------
# TECHNICAL INDICATORS (calculated in native currency, converted for display)
# ----------------------------------------------------------------------------
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["Bollinger_Mid"] = df["Close"].rolling(window=20).mean()
df["Bollinger_Upper"] = df["Bollinger_Mid"] + (df["Close"].rolling(window=20).std() * 2)
df["Bollinger_Lower"] = df["Bollinger_Mid"] - (df["Close"].rolling(window=20).std() * 2)
df["Daily_Return_%"] = df["Close"].pct_change() * 100

# RSI (14-period)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(window=14).mean()
loss = (-delta.clip(upper=0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
df["RSI_14"] = 100 - (100 / (1 + rs))

# Build a display copy with converted price columns (volume stays share-count)
price_cols = ["Open", "High", "Low", "Close", "SMA_50", "SMA_200", "EMA_20",
              "Bollinger_Mid", "Bollinger_Upper", "Bollinger_Lower"]
disp = df.copy()
for col in price_cols:
    disp[col] = disp[col] * fx_rate

# ----------------------------------------------------------------------------
# KEY METRICS
# ----------------------------------------------------------------------------
latest_price = float(disp["Close"].iloc[-1])
prev_price = float(disp["Close"].iloc[-2]) if len(disp) > 1 else latest_price
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100 if prev_price else 0.0

lookback_252 = disp.tail(252)
week52_high = float(lookback_252["High"].max())
week52_low = float(lookback_252["Low"].min())

st.markdown('<div class="ps-section-title">Market Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="ps-sub">{ticker_symbol} · {start_date} → {end_date} · '
    f'Priced in {display_currency}</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Close", fmt(latest_price, display_currency),
            f"{'▲' if price_change >= 0 else '▼'} {price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("Day High", fmt(float(disp['High'].iloc[-1]), display_currency))
col3.metric("Day Low", fmt(float(disp['Low'].iloc[-1]), display_currency))
col4.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
col5.metric("52-Week Range", f"{fmt(week52_low, display_currency)} – {fmt(week52_high, display_currency)}")

# ----------------------------------------------------------------------------
# CANDLESTICK + VOLUME CHART
# ----------------------------------------------------------------------------
st.markdown('<div class="ps-section-title">📊 Price Action & Technical Indicators</div>', unsafe_allow_html=True)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
    subplot_titles=(f"{ticker_symbol} — Candlestick with Moving Averages ({display_currency})", "Volume"),
)
fig.add_trace(go.Candlestick(
    x=disp.index, open=disp["Open"], high=disp["High"], low=disp["Low"], close=disp["Close"],
    name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp["SMA_50"], line=dict(color="#facc15", width=1.4), name="SMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp["SMA_200"], line=dict(color="#38bdf8", width=1.4), name="SMA 200"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp["EMA_20"], line=dict(color="#c084fc", width=1.2, dash="dot"), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp["Bollinger_Upper"], line=dict(color="#6b7280", width=1, dash="dash"), name="Bollinger Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=disp.index, y=disp["Bollinger_Lower"], line=dict(color="#6b7280", width=1, dash="dash"),
                          name="Bollinger Lower", fill="tonexty", fillcolor="rgba(107,114,128,0.08)"), row=1, col=1)

volume_colors = np.where(df["Close"] >= df["Open"], "#22c55e", "#ef4444")
fig.add_trace(go.Bar(x=disp.index, y=df["Volume"], name="Volume", marker_color=volume_colors), row=2, col=1)

fig.update_layout(
    template="plotly_dark", height=650, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="#e6e9ef"), xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ RSI (Relative Strength Index) — momentum indicator"):
    latest_rsi = df["RSI_14"].dropna().iloc[-1] if df["RSI_14"].notna().any() else None
    if latest_rsi is not None:
        zone = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Neutral"
        st.write(f"Current RSI(14): **{latest_rsi:.1f}** — {zone}. RSI measures the speed and "
                 "magnitude of recent price changes; readings above 70 suggest the stock may be "
                 "overbought, below 30 suggest it may be oversold.")
    else:
        st.write("Not enough data points yet to calculate a 14-period RSI.")

# ----------------------------------------------------------------------------
# MACHINE LEARNING — RANDOM FOREST PREDICTION (multi-feature, chronological split)
# ----------------------------------------------------------------------------
st.markdown('<div class="ps-section-title">🤖 AI Price Prediction & Decision Signal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps-sub">Model: Random Forest Regressor · Features: Open, High, Low, Close, '
    'Volume, SMA 50, EMA 20 · Target: next trading day\'s close</div>',
    unsafe_allow_html=True,
)

run_model = st.button("▶ Run AI Prediction")

if run_model:
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "EMA_20"]
    model_df = df[feature_cols].copy()
    model_df["Target"] = model_df["Close"].shift(-1)
    model_df.dropna(inplace=True)

    if len(model_df) > 30:
        X = model_df[feature_cols].values
        y = model_df["Target"].values

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        with st.spinner("Training AI model on historical data..."):
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
        r2 = r2_score(y_test, predictions)

        st.success("Model trained successfully on chronologically-split historical data.")

        # --- Forward forecast: refit on ALL data, predict the next unseen day ---
        final_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        final_model.fit(X, y)
        last_row = df[feature_cols].iloc[[-1]].values
        next_day_pred_native = float(final_model.predict(last_row)[0])
        next_day_pred = next_day_pred_native * fx_rate

        current_close_disp = float(disp["Close"].iloc[-1])
        expected_change = next_day_pred - current_close_disp
        expected_change_pct = (expected_change / current_close_disp) * 100 if current_close_disp else 0.0

        if expected_change_pct > 1.0:
            signal, css_class, arrow = "BUY", "ps-buy", "▲"
        elif expected_change_pct < -1.0:
            signal, css_class, arrow = "SELL", "ps-sell", "▼"
        else:
            signal, css_class, arrow = "HOLD / NEUTRAL", "ps-hold", "▬"

        fcol1, fcol2 = st.columns([1, 1])
        with fcol1:
            st.metric("Next-Day Predicted Close", fmt(next_day_pred, display_currency),
                       f"{arrow} {expected_change:+.2f} ({expected_change_pct:+.2f}%)")
        with fcol2:
            st.markdown(
                f'<div style="padding-top:22px;">AI-Assisted Decision Signal:<br>'
                f'<span class="ps-signal {css_class}">{arrow} {signal}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "This signal is derived only from the model's own next-day estimate versus the "
            "current price (±1% threshold). It is a model output, not investment advice, and "
            "carries no guarantee of accuracy."
        )

        st.markdown("**Model Performance (on held-out test data)**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", fmt(mae * fx_rate, display_currency))
        m2.metric("RMSE", fmt(rmse * fx_rate, display_currency))
        m3.metric("R² Score", f"{r2:.3f}")
        m4.metric("Test Samples", f"{len(y_test)}")

        pred_dates = model_df.index[split_idx:]
        pred_df = pd.DataFrame({
            "Actual": y_test * fx_rate,
            "Predicted": predictions * fx_rate,
        }, index=pred_dates)

        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"],
                                       line=dict(color="#38bdf8", width=2), name="Actual Price"))
        pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predicted"],
                                       line=dict(color="#facc15", width=2, dash="dot"), name="Predicted Price"))
        pred_fig.update_layout(
            template="plotly_dark", height=420, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e6e9ef"), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"Actual vs. Predicted Closing Price — Test Set ({display_currency})",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        pred_fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
        pred_fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
        st.plotly_chart(pred_fig, use_container_width=True)

        csv_data = pred_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
        st.download_button("⬇ Download Prediction Results (CSV)", data=csv_data,
                            file_name=f"{ticker_symbol}_predictions_{display_currency}.csv", mime="text/csv")
    else:
        st.warning("Not enough data points in this range to train the AI model. Try a wider date range.")

# ----------------------------------------------------------------------------
# RAW DATA + FULL EXPORT
# ----------------------------------------------------------------------------
with st.expander("📄 View Raw Market Data Table"):
    display_table = disp[["Open", "High", "Low", "Close", "SMA_50", "SMA_200", "EMA_20", "RSI_14"]].copy()
    display_table["Volume"] = df["Volume"]
    st.dataframe(display_table.tail(50).round(2), use_container_width=True)
    full_csv = display_table.to_csv().encode("utf-8")
    st.download_button("⬇ Download Full Dataset (CSV)", data=full_csv,
                        file_name=f"{ticker_symbol}_raw_data_{display_currency}.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="ps-footer">
        PredictStock Dashboard · Data via Yahoo Finance (yfinance) · FX rates via live currency pairs ·
        Educational/research prototype — not financial advice. No guaranteed accuracy or returns.
    </div>
    """,
    unsafe_allow_html=True,
)