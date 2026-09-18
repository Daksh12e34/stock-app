"""
PredictStock — Institutional-Grade Stock Analysis & Prediction Dashboard
Built with Streamlit, yfinance, Plotly, and scikit-learn.

v6 — Enhanced UI + Developer-Grade ML Accuracy Upgrades
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
# CUSTOM CSS — FINTECH DARK THEME v6 (RGB GLOW + SMOOTH)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0a0d14 !important;
        color: #e6e9ef !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(59,130,246,0.12) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(168,85,247,0.10) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(34,197,94,0.04) 0%, transparent 60%),
            linear-gradient(180deg, #0a0d14 0%, #0e1117 100%);
        background-attachment: fixed;
        color: #e6e9ef;
    }
    #MainMenu, footer {visibility: hidden;}

    /* ============ ANIMATIONS ============ */
    @keyframes pulseDot {
        0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.7); }
        50% { transform: scale(1.15); opacity: 0.85; box-shadow: 0 0 0 8px rgba(74,222,128,0); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes tickerScroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    @keyframes signalPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.45); }
        50% { box-shadow: 0 0 0 12px rgba(74,222,128,0); }
    }
    @keyframes rgbGlow {
        0% { box-shadow: 0 0 20px rgba(59,130,246,0.35), 0 0 40px rgba(168,85,247,0.15); border-color: rgba(59,130,246,0.5); }
        33% { box-shadow: 0 0 20px rgba(168,85,247,0.35), 0 0 40px rgba(34,197,94,0.15); border-color: rgba(168,85,247,0.5); }
        66% { box-shadow: 0 0 20px rgba(34,197,94,0.35), 0 0 40px rgba(59,130,246,0.15); border-color: rgba(34,197,94,0.5); }
        100% { box-shadow: 0 0 20px rgba(59,130,246,0.35), 0 0 40px rgba(168,85,247,0.15); border-color: rgba(59,130,246,0.5); }
    }
    @keyframes rgbBorder {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* ============ HERO HEADER ============ */
    .ps-header {
        position: relative;
        display: flex; justify-content: space-between; align-items: center;
        padding: 26px 32px;
        background: linear-gradient(135deg, #121722 0%, #161b26 50%, #1a1420 100%);
        background-size: 200% 200%;
        animation: gradientShift 14s ease infinite, fadeInUp 0.6s ease;
        border: 1px solid #232a37;
        border-radius: 16px;
        margin-bottom: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .ps-header::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #22c55e, #3b82f6);
        background-size: 300% 100%;
        animation: gradientShift 8s linear infinite;
    }
    .ps-header h1 {
        font-size: 28px; font-weight: 800; margin: 0;
        color: #f8fafc; letter-spacing: -0.4px;
        background: linear-gradient(90deg, #f8fafc, #93c5fd, #c4b5fd, #f8fafc);
        background-size: 200% 100%;
        animation: gradientShift 6s linear infinite;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ps-header p { margin: 4px 0 0 0; font-size: 13.5px; color: #94a3b8; letter-spacing: 0.1px; }
    .ps-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(22,36,26,0.6);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.35);
        padding: 7px 16px; border-radius: 999px;
        font-size: 12.5px; font-weight: 600;
        backdrop-filter: blur(8px);
    }
    .ps-badge::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%;
        background: #4ade80;
        animation: pulseDot 2s ease-in-out infinite;
    }

    /* ============ TICKER TAPE ============ */
    .ps-ticker-wrap {
        overflow: hidden; white-space: nowrap;
        background: #0d1017;
        border: 1px solid #1f2530;
        border-radius: 10px;
        margin-bottom: 20px;
        position: relative;
    }
    .ps-ticker-wrap::before,
    .ps-ticker-wrap::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 40px; z-index: 2; pointer-events: none;
    }
    .ps-ticker-wrap::before { left: 0; background: linear-gradient(90deg, #0d1017, transparent); }
    .ps-ticker-wrap::after { right: 0; background: linear-gradient(-90deg, #0d1017, transparent); }
    .ps-ticker-track {
        display: inline-block;
        padding: 10px 0;
        animation: tickerScroll 40s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #cbd5e1;
    }
    .ps-ticker-track span { margin: 0 26px; }
    .ps-ticker-track .up { color: #4ade80; }
    .ps-ticker-track .down { color: #f87171; }

    /* ============ METRIC CARDS (RGB GLOW) ============ */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(19,23,34,0.9), rgba(15,18,26,0.9));
        border: 1px solid #232a37;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.02);
        transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                    border-color 0.35s ease,
                    box-shadow 0.35s ease;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease;
    }
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; height: 3px; width: 100%;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #22c55e, #3b82f6);
        background-size: 300% 100%;
        opacity: 0.85;
        animation: gradientShift 6s linear infinite;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.01);
        animation: rgbGlow 4s ease infinite;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important; font-size: 12.5px !important;
        font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.6px;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important; font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetricDelta"] { font-size: 12.5px !important; font-weight: 600 !important; }

    /* ============ SIDEBAR ============ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0e15 0%, #0d1017 100%) !important;
        border-right: 1px solid #1f2530;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc; font-size: 12.5px;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 22px; margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1f2530;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }

    /* ============ SECTION TITLES ============ */
    .ps-section-title {
        font-size: 19px; font-weight: 700; color: #f8fafc;
        margin: 30px 0 8px 0;
        padding-left: 14px;
        position: relative;
        letter-spacing: -0.2px;
        display: flex; align-items: center; gap: 10px;
    }
    .ps-section-title::before {
        content: ""; position: absolute; left: 0; top: 4px; bottom: 4px; width: 4px;
        border-radius: 4px;
        background: linear-gradient(180deg, #3b82f6, #a855f7, #22c55e);
        box-shadow: 0 0 12px rgba(59,130,246,0.5);
    }
    .ps-sub { color: #8b93a7; font-size: 13px; margin-bottom: 16px; padding-left: 14px; }

    /* ============ SIGNAL BADGE ============ */
    .ps-signal {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 12px 24px; border-radius: 12px;
        font-weight: 800; font-size: 16px;
        letter-spacing: 0.6px;
        font-family: 'JetBrains Mono', monospace;
        animation: fadeInUp 0.5s ease;
    }
    .ps-buy {
        background: linear-gradient(135deg, rgba(22,36,26,0.95), rgba(15,30,20,0.95));
        color: #4ade80; border: 1px solid rgba(74,222,128,0.4);
        animation: signalPulse 2.5s ease-in-out infinite, fadeInUp 0.5s ease;
    }
    .ps-sell {
        background: linear-gradient(135deg, rgba(42,20,22,0.95), rgba(30,15,17,0.95));
        color: #f87171; border: 1px solid rgba(248,113,113,0.4);
    }
    .ps-hold {
        background: linear-gradient(135deg, rgba(36,31,20,0.95), rgba(28,24,14,0.95));
        color: #facc15; border: 1px solid rgba(250,204,21,0.4);
    }

    /* ============ BUTTONS ============ */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white; border: none; border-radius: 10px;
        padding: 10px 22px; font-weight: 700;
        letter-spacing: 0.4px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(59,130,246,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59,130,246,0.6), 0 0 30px rgba(168,85,247,0.3);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ============ INPUTS ============ */
    input, textarea {
        background-color: #131722 !important;
        color: #e6e9ef !important;
        border: 1px solid #232a37 !important;
        border-radius: 8px !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    input:focus, textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15), 0 0 20px rgba(59,130,246,0.25) !important;
    }

    /* ============ SELECTBOX ============ */
    div[data-baseweb="select"] > div {
        background-color: #131722 !important;
        color: #e6e9ef !important;
        border-color: #232a37 !important;
        border-radius: 8px !important;
        transition: all 0.25s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59,130,246,0.2);
    }

    /* ============ DROPDOWN / PORTAL ============ */
    div[data-baseweb="popover"] { background-color: transparent !important; }
    ul[data-baseweb="menu"], div[data-baseweb="menu"] {
        background-color: #131722 !important;
        color: #e6e9ef !important;
        border: 1px solid #232a37 !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
    }
    li[data-baseweb="menu-item"] { color: #e6e9ef !important; }
    li[data-baseweb="menu-item"]:hover { background-color: #1e2530 !important; }
    div[data-baseweb="calendar"] {
        background-color: #131722 !important;
        color: #e6e9ef !important;
        border-radius: 10px !important;
    }

    /* ============ EXPANDERS ============ */
    details {
        background-color: #131722 !important;
        border: 1px solid #232a37 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    details:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 20px rgba(59,130,246,0.15);
    }
    summary { color: #e6e9ef !important; font-weight: 600; padding: 4px 0; }

    /* ============ DATAFRAME ============ */
    [data-testid="stDataFrame"] {
        background-color: #131722 !important;
        border-radius: 10px;
        border: 1px solid #232a37;
        overflow: hidden;
    }

    /* ============ TOOLTIPS ============ */
    [data-testid="stTooltipIcon"] svg { fill: #8b93a7 !important; }

    /* ============ ALERTS ============ */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }

    /* ============ CHART GLOW WRAPPER ============ */
    [data-testid="stPlotlyChart"] {
        border-radius: 14px;
        padding: 8px;
        background: linear-gradient(135deg, rgba(19,23,34,0.4), rgba(13,16,23,0.4));
        border: 1px solid rgba(35,42,55,0.6);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        transition: all 0.4s ease;
    }
    [data-testid="stPlotlyChart"]:hover {
        border-color: rgba(59,130,246,0.4);
        box-shadow: 0 0 30px rgba(59,130,246,0.2), 0 4px 24px rgba(0,0,0,0.3);
    }

    /* ============ FOOTER ============ */
    .ps-footer {
        margin-top: 50px; padding: 22px 0; border-top: 1px solid #232a37;
        color: #6b7280; font-size: 12px; text-align: center;
        line-height: 1.7;
    }
    .ps-footer-badges {
        display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-top: 10px;
    }
    .ps-footer-badges span {
        background: #131722; border: 1px solid #232a37;
        padding: 4px 12px; border-radius: 999px;
        font-size: 11px; color: #94a3b8; font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.25s ease;
    }
    .ps-footer-badges span:hover {
        border-color: #3b82f6;
        color: #93c5fd;
        box-shadow: 0 0 15px rgba(59,130,246,0.3);
    }

    /* ============ SCROLLBAR ============ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0a0d14; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #232a37, #2f3947);
        border-radius: 6px;
        border: 2px solid #0a0d14;
    }
    ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

    html { scroll-behavior: smooth; }

    div[data-testid="stSpinner"] > div {
        border-top-color: #3b82f6 !important;
    }
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


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_native_currency(ticker: str) -> str:
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
def get_fx_rate(from_currency: str, to_currency: str):
    a = "USD" if from_currency == "USDT" else from_currency
    b = "USD" if to_currency == "USDT" else to_currency
    if a == b:
        return 1.0, True
    for pair_symbol, invert in ((f"{a}{b}=X", False), (f"{b}{a}=X", True)):
        try:
            data = yf.download(pair_symbol, period="5d", progress=False)
            data = _flatten_columns(data)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()
                if len(closes) > 0:
                    rate = float(closes.iloc[-1])
                    return (1.0 / rate, True) if invert else (rate, True)
        except Exception:
            continue
    return 1.0, False


def fmt(value: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}"


# ----------------------------------------------------------------------------
# STOCK SEARCH
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def search_stocks(query: str):
    query = (query or "").strip()
    if len(query) < 1:
        return []
    q_lower = query.lower()
    raw_results = []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0 (compatible; PredictStock/1.0)"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("quotes", []):
            symbol = item.get("symbol")
            if not symbol:
                continue
            name = item.get("shortname") or item.get("longname") or ""
            exch = item.get("exchange", "")
            kind = item.get("quoteType", "")
            raw_results.append((symbol, name, exch, kind))
    except Exception:
        pass
    if not raw_results:
        try:
            s = yf.Search(query, max_results=10)
            for item in getattr(s, "quotes", []):
                symbol = item.get("symbol")
                if not symbol:
                    continue
                name = item.get("shortname") or item.get("longname") or ""
                exch = item.get("exchange", "")
                kind = item.get("quoteType", "")
                raw_results.append((symbol, name, exch, kind))
        except Exception:
            pass

    def relevance(item):
        symbol, name, exch, kind = item
        symbol_l, name_l = symbol.lower(), name.lower()
        symbol_hit = 0 if q_lower in symbol_l else 1
        name_hit = 0 if q_lower in name_l else 1
        is_equity_like = 0 if kind in ("EQUITY", "ETF", "") else 1
        return (min(symbol_hit, name_hit), is_equity_like)

    raw_results.sort(key=relevance)
    labeled = []
    for symbol, name, exch, _ in raw_results[:8]:
        bits = [symbol]
        if name:
            bits.append(f"— {name}")
        if exch:
            bits.append(f"({exch})")
        labeled.append((symbol, " ".join(bits)))
    return labeled


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
        <div class="ps-badge">Live Data · {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# LIVE TICKER TAPE
# ----------------------------------------------------------------------------
_ticker_items = [
    ("NIFTY 50", "+0.62%", "up"), ("SENSEX", "+0.48%", "up"),
    ("NASDAQ", "-0.21%", "down"), ("S&P 500", "+0.15%", "up"),
    ("BTC/USD", "+1.84%", "up"), ("ETH/USD", "-0.52%", "down"),
    ("AAPL", "+0.94%", "up"), ("TSLA", "-1.12%", "down"),
    ("RELIANCE", "+0.33%", "up"), ("INFY", "+1.05%", "up"),
    ("GOLD", "+0.21%", "up"), ("USD/INR", "-0.08%", "down"),
]
_tape_html = "".join(
    f'<span>{name} <b class="{cls}">{chg}</b></span>' for name, chg, cls in _ticker_items
)
st.markdown(
    f"""
    <div class="ps-ticker-wrap">
        <div class="ps-ticker-track">{_tape_html}{_tape_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — MARKET CONTROLS
# ----------------------------------------------------------------------------
st.sidebar.header("Market Controls")

if "ticker_symbol" not in st.session_state:
    st.session_state["ticker_symbol"] = "AAPL"


def _apply_search_pick():
    label = st.session_state.get("_search_pick_label")
    mapping = st.session_state.get("_search_label_map", {})
    if label in mapping:
        st.session_state["ticker_symbol"] = mapping[label]


search_query = st.sidebar.text_input(
    "🔍 Search any stock (name or symbol)",
    value="",
    placeholder="e.g. Amazon, Infosys, Tesla, Reliance, TCS",
    help="Works for any company listed on Yahoo Finance.",
)

search_results = search_stocks(search_query) if search_query.strip() else []

if search_results:
    match_labels = [label for _, label in search_results]
    st.session_state["_search_label_map"] = {label: sym for sym, label in search_results}
    st.sidebar.selectbox(
        "Matching results — select one",
        match_labels,
        key="_search_pick_label",
        on_change=_apply_search_pick,
    )
elif search_query.strip():
    st.sidebar.caption("No matches found. Type the exact ticker below instead.")

ticker_symbol = st.sidebar.text_input(
    "Stock Ticker Symbol", key="ticker_symbol",
    help="Type any symbol yfinance supports.",
).upper().strip()

native_currency = get_native_currency(ticker_symbol)

CURRENCY_OPTIONS = ["USD", "INR", "EUR", "GBP", "JPY", "USDT"]
display_currency = st.sidebar.selectbox(
    "Display Currency",
    CURRENCY_OPTIONS,
    index=CURRENCY_OPTIONS.index(native_currency) if native_currency in CURRENCY_OPTIONS else 0,
)

fx_rate, fx_ok = get_fx_rate(native_currency, display_currency)
if fx_ok:
    st.sidebar.caption(
        f"Native: **{native_currency}** → Display: **{display_currency}** "
        f"(1 {native_currency} = {fx_rate:.4f} {display_currency})"
    )
elif native_currency != display_currency:
    st.sidebar.warning(
        f"⚠️ Couldn't fetch live {native_currency}→{display_currency} rate. "
        f"Showing in native currency."
    )
    display_currency = native_currency
    fx_rate = 1.0

# ----------------------------------------------------------------------------
# SIDEBAR — TIMEFRAME
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
# SIDEBAR — ML MODEL PARAMETERS (v6 with auto-tune)
# ----------------------------------------------------------------------------
st.sidebar.header("ML Model Parameters")
auto_tune = st.sidebar.checkbox(
    "⚡ Auto-tune hyperparameters (slower, more accurate)",
    value=True,
    help="Runs RandomizedSearchCV over the model's hyperparameters to find "
         "the best configuration for THIS stock's data. Adds ~10–20s to "
         "training but typically improves accuracy noticeably.",
)
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 400, 200)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 40, 15)

with st.sidebar.expander("ℹ️ How does the prediction model work?"):
    st.write(
        "A **Random Forest Regressor** learns from each day's Open, High, Low, "
        "Close, Volume, moving averages, **momentum lags, volatility and RSI** "
        "to estimate the *next* day's closing price. Data is split "
        "**chronologically** (no shuffling) so the model is never trained on "
        "the future. With **auto-tune** enabled, the model searches for the "
        "best hyperparameters for this specific stock. This is a short-term "
        "statistical estimate, **not** financial advice."
    )

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    return _flatten_columns(df)


try:
    with st.spinner(f"Fetching live market data for {ticker_symbol}..."):
        df = load_data(ticker_symbol, start_date, end_date)
except Exception as e:
    st.error(f"⚠️ Could not fetch data for '{ticker_symbol}'. Details: {e}")
    st.stop()

if df.empty or "Close" not in df.columns:
    st.error(
        f"⚠️ No market data found for '{ticker_symbol}' in the selected date range."
    )
    st.stop()

if len(df) < 30:
    st.warning(
        "⚠️ Very little historical data — indicators and prediction will be less reliable."
    )

# ----------------------------------------------------------------------------
# TECHNICAL INDICATORS
# ----------------------------------------------------------------------------
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["Bollinger_Mid"] = df["Close"].rolling(window=20).mean()
df["Bollinger_Upper"] = df["Bollinger_Mid"] + (df["Close"].rolling(window=20).std() * 2)
df["Bollinger_Lower"] = df["Bollinger_Mid"] - (df["Close"].rolling(window=20).std() * 2)
df["Daily_Return_%"] = df["Close"].pct_change() * 100

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(window=14).mean()
loss = (-delta.clip(upper=0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# ---- NEW v6 FEATURES (REAL accuracy boosters) ----
df["Lag_Close_1"] = df["Close"].shift(1)
df["Lag_Close_2"] = df["Close"].shift(2)
df["Lag_Close_3"] = df["Close"].shift(3)
df["Volatility_20"] = df["Close"].pct_change().rolling(window=20).std()
df["Return_Lag_1"] = df["Close"].pct_change().shift(1)

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
        st.write(f"Current RSI(14): **{latest_rsi:.1f}** — {zone}.")
    else:
        st.write("Not enough data points yet to calculate a 14-period RSI.")

# ----------------------------------------------------------------------------
# MACHINE LEARNING — v6 with real accuracy upgrades
# ----------------------------------------------------------------------------
st.markdown('<div class="ps-section-title">🤖 AI Price Prediction & Decision Signal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ps-sub">Model: Random Forest Regressor · Features: OHLCV + SMA50 + EMA20 + '
    'Lag features + Volatility + RSI + MACD · Target: next trading day\'s close</div>',
    unsafe_allow_html=True,
)

run_model = st.button("▶ Run AI Prediction")

if run_model:
    # ---- ENHANCED FEATURE SET (v6) ----
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_50", "EMA_20",
        "RSI_14", "MACD", "MACD_Signal",
        "Lag_Close_1", "Lag_Close_2", "Lag_Close_3",
        "Volatility_20", "Return_Lag_1",
    ]
    model_df = df[feature_cols].copy()
    model_df["Target"] = model_df["Close"].shift(-1)
    model_df.dropna(inplace=True)

    if len(model_df) > 60:
        X = model_df[feature_cols].values
        y = model_df["Target"].values

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ---- HYPERPARAMETER TUNING (v6 accuracy upgrade) ----
        if auto_tune and len(X_train) > 80:
            with st.spinner("Auto-tuning hyperparameters (RandomizedSearchCV)..."):
                param_dist = {
                    "n_estimators": [100, 150, 200, 250, 300],
                    "max_depth": [5, 8, 10, 15, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                }
                tscv = TimeSeriesSplit(n_splits=3)
                search = RandomizedSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    param_distributions=param_dist,
                    n_iter=15,
                    cv=tscv,
                    scoring="neg_mean_absolute_error",
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_params = search.best_params_
                st.info(f"🎯 Best hyperparameters found: `{best_params}`")
        else:
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
            }

        with st.spinner("Training AI model on historical data..."):
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
        r2 = r2_score(y_test, predictions)

        # Directional accuracy — % of times it got up/down right
        direction_actual = np.sign(np.diff(y_test))
        direction_pred = np.sign(np.diff(predictions))
        if len(direction_actual) > 0:
            dir_acc = float(np.mean(direction_actual == direction_pred) * 100)
        else:
            dir_acc = 0.0

        st.success("Model trained successfully on chronologically-split historical data.")

        # ---- FORWARD FORECAST with confidence interval ----
        final_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
        final_model.fit(X, y)
        last_row = df[feature_cols].iloc[[-1]].values
        per_tree_preds = np.array([tree.predict(last_row)[0] for tree in final_model.estimators_])
        next_day_pred_native = float(per_tree_preds.mean())
        pred_std_native = float(per_tree_preds.std())
        ci_low_native = next_day_pred_native - 1.96 * pred_std_native
        ci_high_native = next_day_pred_native + 1.96 * pred_std_native

        next_day_pred = next_day_pred_native * fx_rate
        ci_low = ci_low_native * fx_rate
        ci_high = ci_high_native * fx_rate

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
            f"**95% confidence interval:** {fmt(ci_low, display_currency)} — "
            f"{fmt(ci_high, display_currency)} (based on {len(final_model.estimators_)} trees' variance). "
            "This signal is a model output, not investment advice."
        )

        st.markdown("**Model Performance (on held-out test data)**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", fmt(mae * fx_rate, display_currency))
        m2.metric("RMSE", fmt(rmse * fx_rate, display_currency))
        m3.metric("R² Score", f"{r2:.3f}")
        m4.metric("Directional Acc.", f"{dir_acc:.1f}%")
        m5.metric("Test Samples", f"{len(y_test)}")

        # ---- Feature importance ----
        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fi_fig = go.Figure(go.Bar(
            x=importances.values[::-1],
            y=importances.index[::-1],
            orientation="h",
            marker=dict(
                color=importances.values[::-1],
                colorscale=[[0, "#1e3a8a"], [0.5, "#3b82f6"], [1, "#a855f7"]],
            ),
        ))
        fi_fig.update_layout(
            template="plotly_dark", height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e6e9ef"),
            title="Feature Importance — what drives this stock's prediction",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fi_fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
        fi_fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
        st.plotly_chart(fi_fig, use_container_width=True)

        # ---- Actual vs Predicted ----
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
        st.warning("Not enough data points (need at least 60) to train the AI model. Try a wider date range.")

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
        PredictStock Dashboard · Data via Yahoo Finance (yfinance) · FX rates via live currency pairs<br>
        Educational/research prototype — not financial advice. No guaranteed accuracy or returns.
        <div class="ps-footer-badges">
            <span>Python</span><span>Streamlit</span><span>Plotly</span>
            <span>scikit-learn</span><span>yfinance</span><span>Random Forest</span>
            <span>TimeSeriesSplit</span><span>RandomizedSearchCV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
