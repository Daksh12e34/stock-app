"""
PredictStock — Institutional-Grade Stock Analysis & Prediction Dashboard
Built with Streamlit, yfinance, Plotly, and scikit-learn.

Features:
  • Live multi-currency pricing (USD, INR, EUR, GBP, JPY, USDT)
  • Smart stock search (any company, any exchange)
  • Composite AI trading signal (RSI + SMA + Bollinger + momentum)
  • Multi-model ensemble prediction (Random Forest / Gradient Boosting / Linear)
  • News sentiment analysis from Yahoo Finance headlines
  • Risk metrics (volatility, Sharpe, max drawdown)
  • Normalized multi-ticker comparison mode
"""

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0a0d14 !important;
        color: #e6e9ef !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .stApp {
        background:
            radial-gradient(
                1200px 600px at 10% -10%,
                rgba(59,130,246,0.10),
                transparent 60%
            ),
            radial-gradient(
                1000px 500px at 90% 0%,
                rgba(139,92,246,0.08),
                transparent 55%
            ),
            linear-gradient(
                180deg,
                #0a0d14 0%,
                #0e1117 100%
            );
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    .ps-header {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 22px 28px;
        background: linear-gradient(
            120deg,
            #101624 0%,
            #131a2a 50%,
            #101624 100%
        );
        border: 1px solid #1f2a3d;
        border-radius: 18px;
        margin-bottom: 24px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }

    .ps-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -50%;
        width: 200%;
        height: 2px;
        background: linear-gradient(
            90deg,
            transparent,
            #3b82f6,
            #8b5cf6,
            transparent
        );
        animation: shine 4s linear infinite;
    }

    @keyframes shine {
        0% {
            transform: translateX(-50%);
        }

        100% {
            transform: translateX(50%);
        }
    }

    .ps-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(
            90deg,
            #f3f5f9 0%,
            #93c5fd 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.3px;
    }

    .ps-header p {
        margin: 4px 0 0 0;
        font-size: 13px;
        color: #8b93a7;
    }

    .ps-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(74,222,128,0.08);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.25);
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        backdrop-filter: blur(8px);
    }

    .ps-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 0 0 rgba(74,222,128,0.7);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(74,222,128,0.7);
        }

        70% {
            box-shadow: 0 0 0 10px rgba(74,222,128,0);
        }

        100% {
            box-shadow: 0 0 0 0 rgba(74,222,128,0);
        }
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #111726 0%,
            #131a2a 100%
        );
        border: 1px solid #1f2a3d;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(59,130,246,0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #8b93a7 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f5f9 !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px !important;
    }

    div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
    }

    section[data-testid="stSidebar"] {
        background: #0b0f18 !important;
        border-right: 1px solid #1a2233;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f3f5f9;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 20px;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    .ps-section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 18px;
        font-weight: 700;
        color: #f3f5f9;
        margin: 30px 0 8px 0;
    }

    .ps-section-title::before {
        content: '';
        width: 4px;
        height: 22px;
        border-radius: 2px;
        background: linear-gradient(
            180deg,
            #3b82f6,
            #8b5cf6
        );
    }

    .ps-sub {
        color: #8b93a7;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .ps-signal {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 12px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.3px;
        font-family: 'JetBrains Mono', monospace;
    }

    .ps-buy {
        background: linear-gradient(
            135deg,
            #14281b,
            #0f2018
        );
        color: #4ade80;
        border: 1px solid #2a5f3a;
    }

    .ps-sell {
        background: linear-gradient(
            135deg,
            #2a1416,
            #210f11
        );
        color: #f87171;
        border: 1px solid #5c2226;
    }

    .ps-hold {
        background: linear-gradient(
            135deg,
            #241f14,
            #1c1810
        );
        color: #facc15;
        border: 1px solid #5c4f22;
    }

    .ps-glass {
        background: linear-gradient(
            145deg,
            rgba(17,23,38,0.85),
            rgba(19,26,42,0.75)
        );
        border: 1px solid #1f2a3d;
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(12px);
    }

    .ps-news-item {
        padding: 12px 14px;
        margin: 8px 0;
        background: rgba(19,26,42,0.6);
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .ps-news-item:hover {
        background: rgba(30,37,54,0.9);
        transform: translateX(3px);
    }

    .ps-news-pos {
        border-left-color: #4ade80 !important;
    }

    .ps-news-neg {
        border-left-color: #f87171 !important;
    }

    .ps-news-neu {
        border-left-color: #94a3b8 !important;
    }

    .ps-news-title {
        color: #e6e9ef;
        font-size: 13.5px;
        font-weight: 500;
    }

    .ps-news-meta {
        color: #6b7280;
        font-size: 11px;
        margin-top: 4px;
    }

    .ps-footer {
        margin-top: 50px;
        padding: 20px 0;
        border-top: 1px solid #1f2a3d;
        color: #6b7280;
        font-size: 12px;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #3b82f6,
            #6366f1
        );
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(59,130,246,0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59,130,246,0.4);
        background: linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );
    }

    input,
    textarea {
        background-color: #111726 !important;
        color: #e6e9ef !important;
        border: 1px solid #1f2a3d !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #111726 !important;
        color: #e6e9ef !important;
        border-color: #1f2a3d !important;
    }

    div[data-baseweb="popover"] {
        background-color: transparent !important;
    }

    ul[data-baseweb="menu"],
    div[data-baseweb="menu"] {
        background-color: #111726 !important;
        color: #e6e9ef !important;
        border: 1px solid #1f2a3d !important;
    }

    li[data-baseweb="menu-item"] {
        color: #e6e9ef !important;
    }

    li[data-baseweb="menu-item"]:hover {
        background-color: #1e2530 !important;
    }

    div[data-baseweb="calendar"] {
        background-color: #111726 !important;
        color: #e6e9ef !important;
    }

    details {
        background-color: #111726 !important;
        border: 1px solid #1f2a3d !important;
        border-radius: 10px !important;
    }

    summary {
        color: #e6e9ef !important;
    }

    [data-testid="stDataFrame"] {
        background-color: #111726 !important;
    }

    [data-testid="stTooltipIcon"] svg {
        fill: #8b93a7 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0d14;
    }

    ::-webkit-scrollbar-thumb {
        background: #1f2a3d;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #2f3947;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# CURRENCY SUPPORT
# ----------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "USDT": "₮",
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns (single-ticker yfinance quirk) to plain names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    return df


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
def get_fx_rate(from_currency: str, to_currency: str):
    """
    Fetch a live conversion rate.
    USDT is treated as a USD-pegged stablecoin.

    Returns:
        (rate, success_flag)

    success_flag=False when no real rate could be fetched,
    so the caller can warn rather than silently use 1:1.
    """

    a = "USD" if from_currency == "USDT" else from_currency
    b = "USD" if to_currency == "USDT" else to_currency

    if a == b:
        return 1.0, True

    for pair_symbol, invert in (
        (f"{a}{b}=X", False),
        (f"{b}{a}=X", True),
    ):
        try:
            data = yf.download(
                pair_symbol,
                period="5d",
                progress=False
            )

            data = _flatten_columns(data)

            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()

                if len(closes) > 0:
                    rate = float(closes.iloc[-1])

                    if invert:
                        return 1.0 / rate, True

                    return rate, True

        except Exception:
            continue

    return 1.0, False


def fmt(value: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(
        currency,
        currency + " "
    )

    return f"{symbol}{value:,.2f}"


# ----------------------------------------------------------------------------
# STOCK SEARCH
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=1800)
def search_stocks(query: str):
    """Look up matching tickers via Yahoo Finance search."""

    query = (query or "").strip()

    if len(query) < 1:
        return []

    q_lower = query.lower()
    raw_results = []

    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={
                "q": query,
                "quotesCount": 10,
                "newsCount": 0,
            },
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PredictStock/1.0)"
            },
            timeout=5,
        )

        resp.raise_for_status()

        data = resp.json()

        for item in data.get("quotes", []):
            symbol = item.get("symbol")

            if not symbol:
                continue

            name = (
                item.get("shortname")
                or item.get("longname")
                or ""
            )

            exch = item.get("exchange", "")
            kind = item.get("quoteType", "")

            raw_results.append(
                (symbol, name, exch, kind)
            )

    except Exception:
        pass

    if not raw_results:
        try:
            s = yf.Search(
                query,
                max_results=10
            )

            for item in getattr(s, "quotes", []):
                symbol = item.get("symbol")

                if not symbol:
                    continue

                name = (
                    item.get("shortname")
                    or item.get("longname")
                    or ""
                )

                exch = item.get("exchange", "")
                kind = item.get("quoteType", "")

                raw_results.append(
                    (symbol, name, exch, kind)
                )

        except Exception:
            pass

    def relevance(item):
        symbol, name, exch, kind = item

        symbol_l = symbol.lower()
        name_l = name.lower()

        symbol_hit = 0 if q_lower in symbol_l else 1
        name_hit = 0 if q_lower in name_l else 1

        is_equity_like = (
            0
            if kind in ("EQUITY", "ETF", "")
            else 1
        )

        return (
            min(symbol_hit, name_hit),
            is_equity_like
        )

    raw_results.sort(key=relevance)

    labeled = []

    for symbol, name, exch, _ in raw_results[:8]:
        bits = [symbol]

        if name:
            bits.append(f"— {name}")

        if exch:
            bits.append(f"({exch})")

        labeled.append(
            (symbol, " ".join(bits))
        )

    return labeled


# ----------------------------------------------------------------------------
# SENTIMENT ANALYSIS
# ----------------------------------------------------------------------------

POSITIVE_WORDS = {
    "surge",
    "surges",
    "soar",
    "soars",
    "rally",
    "rallies",
    "jump",
    "jumps",
    "gain",
    "gains",
    "beat",
    "beats",
    "record",
    "high",
    "upgrade",
    "upgrades",
    "bullish",
    "outperform",
    "profit",
    "profits",
    "growth",
    "boost",
    "strong",
    "buy",
    "buyback",
    "dividend",
    "raises",
    "raised",
    "positive",
    "win",
    "wins",
}

NEGATIVE_WORDS = {
    "drop",
    "drops",
    "fall",
    "falls",
    "plunge",
    "plunges",
    "sink",
    "sinks",
    "crash",
    "crashes",
    "loss",
    "losses",
    "miss",
    "misses",
    "low",
    "downgrade",
    "downgrades",
    "bearish",
    "underperform",
    "weak",
    "cut",
    "cuts",
    "layoff",
    "layoffs",
    "lawsuit",
    "investigation",
    "fine",
    "penalty",
    "negative",
    "warning",
    "warns",
    "concern",
    "concerns",
    "slump",
    "decline",
}


@st.cache_data(show_spinner=False, ttl=1800)
def get_news_sentiment(ticker: str):
    """Fetch recent headlines and score sentiment by keyword matching."""

    try:
        t = yf.Ticker(ticker)
        news = t.news or []

    except Exception:
        news = []

    results = []
    pos_count = 0
    neg_count = 0

    for item in news[:10]:

        content = item.get("content", item)

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )

        publisher = (
            (content.get("provider") or {}).get("displayName")
            or item.get("publisher")
            or ""
        )

        link = (
            (content.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
            or "#"
        )

        if not title:
            continue

        words = set(
            title
            .lower()
            .replace(",", " ")
            .replace(".", " ")
            .split()
        )

        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)

        if pos > neg:
            label = "positive"
            pos_count += 1

        elif neg > pos:
            label = "negative"
            neg_count += 1

        else:
            label = "neutral"

        results.append(
            {
                "title": title,
                "publisher": publisher,
                "link": link,
                "label": label,
            }
        )

    total = pos_count + neg_count

    score = (
        (pos_count - neg_count) / total
        if total > 0
        else 0
    )

    return results, score


# ----------------------------------------------------------------------------
# COMPOSITE TRADING SIGNAL
# ----------------------------------------------------------------------------

def compute_signal(df: pd.DataFrame):
    """Combine RSI + SMA + Bollinger + momentum into BUY/SELL/HOLD."""

    reasons = []
    score = 0

    last = df.iloc[-1]

    rsi = last.get("RSI_14")

    if pd.notna(rsi):

        if rsi < 30:
            score += 1
            reasons.append(
                f"RSI {rsi:.1f} → oversold (bullish)"
            )

        elif rsi > 70:
            score -= 1
            reasons.append(
                f"RSI {rsi:.1f} → overbought (bearish)"
            )

        else:
            reasons.append(
                f"RSI {rsi:.1f} → neutral"
            )

    close = last["Close"]

    sma50 = last.get("SMA_50")
    sma200 = last.get("SMA_200")

    if pd.notna(sma50):

        if close > sma50:
            score += 1
            reasons.append(
                "Price above SMA-50 (uptrend)"
            )

        else:
            score -= 1
            reasons.append(
                "Price below SMA-50 (downtrend)"
            )

    if pd.notna(sma200):

        if close > sma200:
            score += 1
            reasons.append(
                "Price above SMA-200 (long-term bullish)"
            )

        else:
            score -= 1
            reasons.append(
                "Price below SMA-200 (long-term bearish)"
            )

    if pd.notna(sma50) and pd.notna(sma200):

        if sma50 > sma200:
            reasons.append(
                "Golden cross zone (SMA-50 > SMA-200)"
            )

        else:
            reasons.append(
                "Death cross zone (SMA-50 < SMA-200)"
            )

    upper_band = last.get("Bollinger_Upper")
    lower_band = last.get("Bollinger_Lower")

    if pd.notna(upper_band) and pd.notna(lower_band):

        if close <= lower_band:
            score += 1
            reasons.append(
                "Near lower Bollinger Band (potential bounce)"
            )

        elif close >= upper_band:
            score -= 1
            reasons.append(
                "Near upper Bollinger Band (potential pullback)"
            )

    if len(df) > 6:

        ret5 = (
            df["Close"].iloc[-1]
            / df["Close"].iloc[-6]
            - 1
        ) * 100

        if ret5 > 3:
            score += 1
            reasons.append(
                f"Strong 5-day momentum (+{ret5:.1f}%)"
            )

        elif ret5 < -3:
            score -= 1
            reasons.append(
                f"Weak 5-day momentum ({ret5:.1f}%)"
            )

    if score >= 2:
        return "BUY", reasons, score

    elif score <= -2:
        return "SELL", reasons, score

    return "HOLD", reasons, score


# ----------------------------------------------------------------------------
# RISK METRICS
# ----------------------------------------------------------------------------

def compute_risk_metrics(df: pd.DataFrame) -> dict:

    returns = df["Close"].pct_change().dropna()

    if len(returns) < 2:
        return {
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    volatility = (
        returns.std()
        * np.sqrt(252)
        * 100
    )

    rf_daily = 0.05 / 252

    excess = returns - rf_daily

    sharpe = (
        excess.mean()
        / excess.std()
        * np.sqrt(252)
        if excess.std() > 0
        else 0.0
    )

    cum = (1 + returns).cumprod()

    peak = cum.cummax()

    drawdown = (
        cum - peak
    ) / peak

    max_dd = drawdown.min() * 100

    return {
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------

now_str = datetime.datetime.now().strftime(
    "%b %d, %Y — %H:%M"
)

st.markdown(
    f"""
    <div class="ps-header">
        <div>
            <h1>📈 PredictStock</h1>
            <p>
                AI-Powered Market Analysis ·
                Multi-Currency ·
                Sentiment ·
                Risk Metrics
            </p>
        </div>

        <div class="ps-badge">
            <span class="ps-pulse"></span>
            Live · {now_str}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
