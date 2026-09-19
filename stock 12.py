"""
PredictStock — AI Stock Analysis, Prediction, Quiz Game & Login System
With animated emojis, streaks, combos, achievements & beginner feedback.
"""

import datetime
import json
import os
import random

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

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="PredictStock | AI Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

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
    @keyframes floatUp {
        0% { transform: translateY(0) scale(0.5); opacity: 0; }
        20% { transform: translateY(-20px) scale(1.3); opacity: 1; }
        80% { transform: translateY(-100px) scale(1.1); opacity: 0.85; }
        100% { transform: translateY(-150px) scale(0.9); opacity: 0; }
    }
    @keyframes emojiPop {
        0% { transform: scale(0) rotate(-30deg); opacity: 0; }
        50% { transform: scale(1.4) rotate(10deg); opacity: 1; }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-8px); }
        40%, 80% { transform: translateX(8px); }
    }
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        60% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes scorePop {
        0% { transform: scale(1); }
        50% { transform: scale(1.4); color: #facc15; }
        100% { transform: scale(1); }
    }
    @keyframes fireFlicker {
        0%, 100% { transform: scale(1) rotate(-3deg); filter: brightness(1); }
        50% { transform: scale(1.15) rotate(3deg); filter: brightness(1.4); }
    }
    @keyframes glowPulse {
        0%, 100% { filter: drop-shadow(0 0 8px rgba(59,130,246,0.6)); }
        50% { filter: drop-shadow(0 0 20px rgba(168,85,247,0.9)); }
    }
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes confettiFall {
        0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
        100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
    }

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
        background: linear-gradient(90deg, #f8fafc, #93c5fd, #c4b5fd, #f8fafc);
        background-size: 200% 100%;
        animation: gradientShift 6s linear infinite;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ps-header p { margin: 4px 0 0 0; font-size: 13.5px; color: #94a3b8; }
    .ps-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(22,36,26,0.6); color: #4ade80;
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

    .ps-ticker-wrap {
        overflow: hidden; white-space: nowrap;
        background: #0d1017; border: 1px solid #1f2530;
        border-radius: 10px; margin-bottom: 20px; position: relative;
    }
    .ps-ticker-wrap::before, .ps-ticker-wrap::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 40px; z-index: 2; pointer-events: none;
    }
    .ps-ticker-wrap::before { left: 0; background: linear-gradient(90deg, #0d1017, transparent); }
    .ps-ticker-wrap::after { right: 0; background: linear-gradient(-90deg, #0d1017, transparent); }
    .ps-ticker-track {
        display: inline-block; padding: 10px 0;
        animation: tickerScroll 40s linear infinite;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px; color: #cbd5e1;
    }
    .ps-ticker-track span { margin: 0 26px; }
    .ps-ticker-track .up { color: #4ade80; }
    .ps-ticker-track .down { color: #f87171; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(19,23,34,0.9), rgba(15,18,26,0.9));
        border: 1px solid #232a37; padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.35s ease, border-color 0.35s, box-shadow 0.35s;
        position: relative; overflow: hidden;
        animation: fadeInUp 0.5s ease;
    }
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; top: 0; left: 0; height: 3px; width: 100%;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #22c55e, #3b82f6);
        background-size: 300% 100%; opacity: 0.85;
        animation: gradientShift 6s linear infinite;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-4px) scale(1.01); animation: rgbGlow 4s ease infinite; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 12.5px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.6px; }
    div[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0e15 0%, #0d1017 100%) !important;
        border-right: 1px solid #1f2530;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc; font-size: 12.5px;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 22px; margin-bottom: 10px; padding-bottom: 6px;
        border-bottom: 1px solid #1f2530; font-weight: 700;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }

    .ps-section-title {
        font-size: 19px; font-weight: 700; color: #f8fafc;
        margin: 30px 0 8px 0; padding-left: 14px;
        position: relative; letter-spacing: -0.2px;
    }
    .ps-section-title::before {
        content: ""; position: absolute; left: 0; top: 4px; bottom: 4px; width: 4px;
        border-radius: 4px;
        background: linear-gradient(180deg, #3b82f6, #a855f7, #22c55e);
        box-shadow: 0 0 12px rgba(59,130,246,0.5);
    }
    .ps-sub { color: #8b93a7; font-size: 13px; margin-bottom: 16px; padding-left: 14px; }

    .ps-signal {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 12px 24px; border-radius: 12px;
        font-weight: 800; font-size: 16px; letter-spacing: 0.6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .ps-buy { background: linear-gradient(135deg, rgba(22,36,26,0.95), rgba(15,30,20,0.95)); color: #4ade80; border: 1px solid rgba(74,222,128,0.4); animation: signalPulse 2.5s ease-in-out infinite; }
    .ps-sell { background: linear-gradient(135deg, rgba(42,20,22,0.95), rgba(30,15,17,0.95)); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }
    .ps-hold { background: linear-gradient(135deg, rgba(36,31,20,0.95), rgba(28,24,14,0.95)); color: #facc15; border: 1px solid rgba(250,204,21,0.4); }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white; border: none; border-radius: 10px;
        padding: 10px 22px; font-weight: 700; letter-spacing: 0.4px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(59,130,246,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59,130,246,0.6), 0 0 30px rgba(168,85,247,0.3);
    }

    input, textarea {
        background-color: #131722 !important; color: #e6e9ef !important;
        border: 1px solid #232a37 !important; border-radius: 8px !important;
    }
    input:focus, textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #131722 !important; color: #e6e9ef !important;
        border-color: #232a37 !important; border-radius: 8px !important;
    }
    ul[data-baseweb="menu"], div[data-baseweb="menu"] {
        background-color: #131722 !important; color: #e6e9ef !important;
        border: 1px solid #232a37 !important;
    }
    details { background-color: #131722 !important; border: 1px solid #232a37 !important; border-radius: 10px !important; }
    summary { color: #e6e9ef !important; font-weight: 600; }

    [data-testid="stPlotlyChart"] {
        border-radius: 14px; padding: 8px;
        background: linear-gradient(135deg, rgba(19,23,34,0.4), rgba(13,16,23,0.4));
        border: 1px solid rgba(35,42,55,0.6);
    }

    .ps-footer {
        margin-top: 50px; padding: 22px 0; border-top: 1px solid #232a37;
        color: #6b7280; font-size: 12px; text-align: center;
    }
    .ps-footer-badges { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .ps-footer-badges span {
        background: #131722; border: 1px solid #232a37;
        padding: 4px 12px; border-radius: 999px;
        font-size: 11px; color: #94a3b8; font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0a0d14; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #232a37, #2f3947); border-radius: 6px; }

    [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(3):last-child) > [data-testid="column"]:nth-child(1) button {
        background: linear-gradient(135deg, #22c55e, #15803d) !important;
        border: 2px solid rgba(74,222,128,0.7) !important;
        box-shadow: 0 8px 24px rgba(34,197,94,0.35) !important;
        font-size: 20px !important; font-weight: 900 !important;
        padding: 22px 12px !important; color: #fff !important;
        letter-spacing: 1.5px !important; text-transform: uppercase !important;
        border-radius: 14px !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(3):last-child) > [data-testid="column"]:nth-child(2) button {
        background: linear-gradient(135deg, #ef4444, #991b1b) !important;
        border: 2px solid rgba(248,113,113,0.7) !important;
        box-shadow: 0 8px 24px rgba(239,68,68,0.35) !important;
        font-size: 20px !important; font-weight: 900 !important;
        padding: 22px 12px !important; color: #fff !important;
        letter-spacing: 1.5px !important; text-transform: uppercase !important;
        border-radius: 14px !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(3):last-child) > [data-testid="column"]:nth-child(3) button {
        background: linear-gradient(135deg, #f59e0b, #b45309) !important;
        border: 2px solid rgba(250,204,21,0.7) !important;
        box-shadow: 0 8px 24px rgba(245,158,11,0.35) !important;
        font-size: 20px !important; font-weight: 900 !important;
        padding: 22px 12px !important; color: #fff !important;
        letter-spacing: 1.5px !important; text-transform: uppercase !important;
        border-radius: 14px !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(3):last-child) > [data-testid="column"] button:hover {
        transform: translateY(-6px) scale(1.03) !important;
        filter: brightness(1.15);
    }

    .ps-player-card {
        display: flex; align-items: center; gap: 16px;
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(168,85,247,0.1));
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 16px; padding: 18px 24px; margin-bottom: 16px;
        animation: fadeInUp 0.5s ease;
        position: relative; overflow: hidden;
    }
    .ps-player-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #22c55e, #3b82f6);
        background-size: 300% 100%;
        animation: gradientShift 6s linear infinite;
    }
    .ps-player-avatar {
        width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #a855f7);
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; color: white; font-weight: 900;
        box-shadow: 0 0 20px rgba(59,130,246,0.5);
        animation: glowPulse 3s ease-in-out infinite;
    }
    .ps-player-name { font-size: 20px; font-weight: 800; color: #f8fafc; }
    .ps-player-meta { font-size: 13px; color: #94a3b8; margin-top: 2px; }
    .ps-score-chip {
        display: inline-flex; align-items: center; gap: 8px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white; font-weight: 800; font-size: 18px;
        padding: 10px 22px; border-radius: 999px;
        box-shadow: 0 6px 20px rgba(34,197,94,0.4);
        font-family: 'JetBrains Mono', monospace;
    }

    .ps-xp-wrap {
        background: #131722; border: 1px solid #232a37;
        border-radius: 999px; height: 14px; overflow: hidden;
        margin: 10px 0; position: relative;
    }
    .ps-xp-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #22c55e);
        background-size: 200% 100%; border-radius: 999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        animation: gradientShift 4s linear infinite;
        box-shadow: 0 0 12px rgba(168,85,247,0.6);
    }
    .ps-xp-label {
        display: flex; justify-content: space-between;
        font-size: 11px; color: #94a3b8;
        font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;
    }

    .ps-hud-row { display: flex; gap: 12px; margin: 14px 0; flex-wrap: wrap; }
    .ps-hud-chip {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 16px; border-radius: 999px;
        background: linear-gradient(135deg, rgba(19,23,34,0.9), rgba(15,18,26,0.9));
        border: 1px solid #232a37;
        font-weight: 700; font-size: 14px;
        font-family: 'JetBrains Mono', monospace;
    }
    .ps-hud-chip.streak { border-color: rgba(249,115,22,0.5); color: #fb923c; box-shadow: 0 0 20px rgba(249,115,22,0.2); }
    .ps-hud-chip.streak.active { animation: fireFlicker 0.8s ease infinite; }
    .ps-hud-chip.combo { border-color: rgba(168,85,247,0.5); color: #c084fc; box-shadow: 0 0 20px rgba(168,85,247,0.2); }
    .ps-hud-chip.lives { border-color: rgba(239,68,68,0.5); color: #f87171; }
    .ps-hud-chip.hint { border-color: rgba(250,204,21,0.5); color: #facc15; }

    .ps-round-dots { display: flex; gap: 10px; justify-content: center; margin: 14px 0; }
    .ps-round-dot {
        width: 20px; height: 20px; border-radius: 50%;
        background: #1f2530; border: 2px solid #2f3947;
        transition: all 0.4s ease;
    }
    .ps-round-dot.done { background: linear-gradient(135deg, #3b82f6, #a855f7); border-color: #a855f7; box-shadow: 0 0 14px rgba(168,85,247,0.6); }
    .ps-round-dot.wrong { background: linear-gradient(135deg, #ef4444, #991b1b); border-color: #ef4444; box-shadow: 0 0 14px rgba(239,68,68,0.6); }
    .ps-round-dot.active { background: #facc15; border-color: #facc15; box-shadow: 0 0 18px rgba(250,204,21,0.8); animation: pulseDot 1.5s ease-in-out infinite; }

    .ps-feedback {
        text-align: center; padding: 28px; border-radius: 18px;
        font-size: 42px; font-weight: 900; letter-spacing: 2px;
        animation: bounceIn 0.6s ease; margin: 16px 0;
        position: relative;
    }
    .ps-feedback.correct {
        background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.15));
        border: 2px solid rgba(74,222,128,0.5);
        color: #4ade80;
        box-shadow: 0 0 40px rgba(34,197,94,0.3);
    }
    .ps-feedback.wrong {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(153,27,27,0.15));
        border: 2px solid rgba(248,113,113,0.5);
        color: #f87171;
        box-shadow: 0 0 40px rgba(239,68,68,0.3);
        animation: shake 0.6s ease;
    }

    .ps-emoji-container {
        position: fixed; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none; z-index: 9999;
    }
    .ps-floating-emoji {
        position: absolute; font-size: 60px;
        animation: floatUp 2s ease-out forwards;
        filter: drop-shadow(0 0 20px rgba(255,255,255,0.5));
    }
    .ps-reaction-emoji {
        display: inline-block; font-size: 48px;
        animation: emojiPop 0.6s ease;
        filter: drop-shadow(0 0 15px rgba(255,255,255,0.4));
    }

    .ps-result-card {
        background: linear-gradient(135deg, rgba(19,23,34,0.9), rgba(15,18,26,0.9));
        border: 1px solid #232a37; border-radius: 14px;
        padding: 18px 22px; margin: 12px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px; color: #cbd5e1;
    }
    .ps-result-card b { color: #f8fafc; }

    .ps-teach-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.10), rgba(168,85,247,0.06));
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 16px; padding: 22px 26px; margin: 16px 0;
        animation: fadeInUp 0.5s ease;
    }
    .ps-teach-box h4 { color: #93c5fd; margin: 0 0 12px 0; font-size: 17px; font-weight: 800; }
    .ps-teach-box p, .ps-teach-box li { color: #cbd5e1; font-size: 14px; line-height: 1.8; }

    .ps-lesson-card {
        background: linear-gradient(135deg, rgba(19,23,34,0.85), rgba(15,18,26,0.85));
        border: 1px solid rgba(35,42,55,0.8);
        border-radius: 14px; padding: 20px 24px; margin: 14px 0;
    }
    .ps-lesson-card h4 { color: #f8fafc; margin: 0 0 12px 0; font-size: 16px; font-weight: 800; }
    .ps-lesson-card p, .ps-lesson-card li { color: #cbd5e1; font-size: 13.5px; line-height: 1.8; }
    .ps-lesson-card b { color: #f8fafc; }

    .ps-tip-box { background: linear-gradient(135deg, rgba(250,204,21,0.10), rgba(245,158,11,0.05)); border-left: 4px solid #facc15; border-radius: 10px; padding: 14px 18px; margin: 12px 0; color: #fef3c7; font-size: 14px; line-height: 1.7; }
    .ps-tip-box b { color: #fde047; }
    .ps-warn-box { background: linear-gradient(135deg, rgba(239,68,68,0.10), rgba(153,27,27,0.05)); border-left: 4px solid #f87171; border-radius: 10px; padding: 14px 18px; margin: 12px 0; color: #fecaca; font-size: 14px; line-height: 1.7; }
    .ps-warn-box b { color: #fca5a5; }
    .ps-good-box { background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(22,163,74,0.05)); border-left: 4px solid #4ade80; border-radius: 10px; padding: 14px 18px; margin: 12px 0; color: #bbf7d0; font-size: 14px; line-height: 1.7; }
    .ps-good-box b { color: #86efac; }

    .ps-achievement-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
    .ps-achievement {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 8px 14px; border-radius: 10px;
        background: linear-gradient(135deg, rgba(250,204,21,0.15), rgba(245,158,11,0.08));
        border: 1px solid rgba(250,204,21,0.4);
        font-size: 12.5px; font-weight: 700; color: #fde047;
        animation: bounceIn 0.5s ease;
    }

    .ps-hint-box {
        background: linear-gradient(135deg, rgba(250,204,21,0.12), rgba(245,158,11,0.06));
        border: 1px solid rgba(250,204,21,0.4);
        border-radius: 12px; padding: 16px 20px; margin: 12px 0;
        color: #fef3c7; font-size: 14px; line-height: 1.7;
        animation: slideInLeft 0.5s ease;
    }
    .ps-hint-box b { color: #facc15; }

    .ps-confetti { position: fixed; top: -20px; font-size: 28px; animation: confettiFall 3s linear forwards; pointer-events: none; z-index: 9998; }

    .ps-lb-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 18px; background: linear-gradient(135deg, rgba(19,23,34,0.7), rgba(15,18,26,0.7)); border: 1px solid #232a37; border-radius: 10px; margin-bottom: 8px; transition: all 0.25s ease; }
    .ps-lb-row:hover { border-color: #3b82f6; box-shadow: 0 0 20px rgba(59,130,246,0.2); transform: translateX(4px); }
    .ps-lb-row.gold { border-color: rgba(250,204,21,0.6); background: linear-gradient(135deg, rgba(250,204,21,0.1), rgba(19,23,34,0.7)); }
    .ps-lb-row.silver { border-color: rgba(203,213,225,0.4); }
    .ps-lb-row.bronze { border-color: rgba(217,119,6,0.5); }
    .ps-lb-name { font-weight: 700; color: #f8fafc; }
    .ps-lb-score { font-family: 'JetBrains Mono', monospace; color: #4ade80; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CONSTANTS
# ============================================================================
USERS = {
    "demo": "demo123", "admin": "admin123",
    "player1": "pass1", "player2": "pass2",
}
SCORES_FILE = "scores.json"
CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "USDT": "₮"}
CURRENCY_OPTIONS = ["USD", "INR", "EUR", "GBP", "JPY", "USDT"]

QUIZ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
]
ROUNDS_PER_GAME = 5
CHART_DAYS = 30
MAX_LIVES = 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_scores():
    if not os.path.exists(SCORES_FILE):
        return {}
    try:
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_scores(data):
    try:
        with open(SCORES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_user_score(username):
    data = load_scores()
    return data.get(username, {"score": 0, "games": 0, "correct": 0, "total": 0, "best_streak": 0})


def add_game_result(username, correct, total, best_streak=0):
    data = load_scores()
    user = data.get(username, {"score": 0, "games": 0, "correct": 0, "total": 0, "best_streak": 0})
    user["score"] += correct
    user["games"] += 1
    user["correct"] += correct
    user["total"] += total
    if best_streak > user.get("best_streak", 0):
        user["best_streak"] = best_streak
    data[username] = user
    save_scores(data)
    return user


def get_leaderboard():
    data = load_scores()
    board = [(u, info.get("score", 0)) for u, info in data.items()]
    board.sort(key=lambda x: x[1], reverse=True)
    return board


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_native_currency(ticker):
    try:
        info = yf.Ticker(ticker).fast_info
        cur = getattr(info, "currency", None)
        if cur:
            return cur.upper()
    except Exception:
        pass
    return "USD"


@st.cache_data(show_spinner=False, ttl=1800)
def get_fx_rate(from_currency, to_currency):
    a = "USD" if from_currency == "USDT" else from_currency
    b = "USD" if to_currency == "USDT" else to_currency
    if a == b:
        return 1.0, True
    for pair, invert in ((f"{a}{b}=X", False), (f"{b}{a}=X", True)):
        try:
            data = yf.download(pair, period="5d", progress=False)
            data = _flatten_columns(data)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].dropna()
                if len(closes) > 0:
                    rate = float(closes.iloc[-1])
                    return (1.0 / rate, True) if invert else (rate, True)
        except Exception:
            continue
    return 1.0, False


def fmt(value, currency):
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}"


@st.cache_data(show_spinner=False, ttl=1800)
def search_stocks(query):
    query = (query or "").strip()
    if len(query) < 1:
        return []
    q_lower = query.lower()
    raw_results = []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp.raise_for_status()
        for item in resp.json().get("quotes", []):
            symbol = item.get("symbol")
            if not symbol:
                continue
            raw_results.append((
                symbol,
                item.get("shortname") or item.get("longname") or "",
                item.get("exchange", ""),
                item.get("quoteType", ""),
            ))
    except Exception:
        pass

    def relevance(item):
        symbol, name, exch, kind = item
        symbol_l, name_l = symbol.lower(), name.lower()
        exact = 0 if symbol_l == q_lower else 1
        starts = 0 if symbol_l.startswith(q_lower) else 1
        in_symbol = 0 if q_lower in symbol_l else 1
        in_name = 0 if q_lower in name_l else 1
        is_equity_like = 0 if kind == "EQUITY" else 1
        return (exact, starts, in_symbol, in_name, is_equity_like)

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


def fetch_quiz_round():
    for _ in range(10):
        ticker = random.choice(QUIZ_TICKERS)
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if df.empty or len(df) < CHART_DAYS + 5:
                continue
            df = _flatten_columns(df)
            max_start = len(df) - CHART_DAYS - 1
            if max_start < 1:
                continue
            start_idx = random.randint(0, max_start)
            window = df.iloc[start_idx:start_idx + CHART_DAYS]
            next_row = df.iloc[start_idx + CHART_DAYS]

            current_close = float(window["Close"].iloc[-1])
            actual_next = float(next_row["Close"])
            change_pct = (actual_next - current_close) / current_close * 100

            if change_pct > 1.0:
                direction = "UP"
            elif change_pct < -1.0:
                direction = "DOWN"
            else:
                direction = "FLAT"

            return {
                "ticker": ticker, "df": window,
                "current_close": current_close, "actual_next": actual_next,
                "actual_direction": direction, "change_pct": change_pct,
            }
        except Exception:
            continue
    return None


def score_guess(guess, actual_direction):
    guess = guess.upper()
    if guess == "BUY" and actual_direction == "UP":
        return True
    if guess == "SELL" and actual_direction == "DOWN":
        return True
    if guess == "HOLD" and actual_direction == "FLAT":
        return True
    return False


def generate_hint(current):
    chart_df = current["df"]
    closes = chart_df["Close"].values
    first_close = float(closes[0])
    last_close = float(closes[-1])
    trend_pct = (last_close - first_close) / first_close * 100

    highs = chart_df["High"].values
    lows = chart_df["Low"].values
    first_half_high = float(highs[:len(highs)//2].max())
    second_half_high = float(highs[len(highs)//2:].max())
    first_half_low = float(lows[:len(lows)//2].min())
    second_half_low = float(lows[len(lows)//2:].min())

    if trend_pct > 3 and second_half_high > first_half_high and second_half_low > first_half_low:
        return "🔥 The chart shows **higher highs AND higher lows** — that's a textbook **UPTREND**. Buyers are in control."
    elif trend_pct < -3 and second_half_high < first_half_high and second_half_low < first_half_low:
        return "📉 The chart shows **lower highs AND lower lows** — that's a **DOWNTREND**. Sellers are dominating."
    else:
        return "↔️ The price is bouncing in a **range** — no clear trend. The market is undecided."


def generate_quiz_feedback(user_guess, current):
    guess = user_guess.upper()
    actual_dir = current["actual_direction"]
    chart_df = current["df"]

    closes = chart_df["Close"].values
    first_close = float(closes[0])
    last_close = float(closes[-1])
    trend_pct = (last_close - first_close) / first_close * 100

    highs = chart_df["High"].values
    lows = chart_df["Low"].values
    first_half_high = float(highs[:len(highs)//2].max())
    second_half_high = float(highs[len(highs)//2:].max())
    first_half_low = float(lows[:len(lows)//2].min())
    second_half_low = float(lows[len(lows)//2:].min())

    if trend_pct > 3 and second_half_high > first_half_high and second_half_low > first_half_low:
        trend_desc = "a clear **UPTREND** (higher highs + higher lows)"
        expected_signal = "BUY"
        trend_color = "#4ade80"
    elif trend_pct < -3 and second_half_high < first_half_high and second_half_low < first_half_low:
        trend_desc = "a clear **DOWNTREND** (lower highs + lower lows)"
        expected_signal = "SELL"
        trend_color = "#f87171"
    else:
        trend_desc = "a **SIDEWAYS** pattern (bouncing in a range)"
        expected_signal = "HOLD"
        trend_color = "#facc15"

    if guess == "BUY" and actual_dir == "DOWN":
        lesson = "You chose **BUY** but the price **went DOWN**. This is the most common beginner mistake — buying into a falling stock hoping it bounces. When you see **lower highs and lower lows**, sellers are in control."
        warning = "**Pro tip:** Never BUY just because the price \"looks cheap\". Wait for the chart to show **higher lows** before entering."
    elif guess == "BUY" and actual_dir == "FLAT":
        lesson = "You chose **BUY** but the price **stayed flat**. In a sideways market, buyers and sellers are equally strong — no breakout."
        warning = "**Pro tip:** Sideways markets are the hardest to trade. **HOLD** is the smart move."
    elif guess == "SELL" and actual_dir == "UP":
        lesson = "You chose **SELL** but the price **went UP**. You bet against an uptrend. When buyers push to **higher highs**, selling is fighting the trend."
        warning = "**Pro tip:** \"The trend is your friend.\" Don't bet against a strong uptrend."
    elif guess == "SELL" and actual_dir == "FLAT":
        lesson = "You chose **SELL** but the price **stayed flat**. There was no downward pressure. HOLD was safer here."
        warning = "**Pro tip:** Selling in a quiet market usually means paying commissions for nothing."
    elif guess == "HOLD" and actual_dir == "UP":
        lesson = "You chose **HOLD** but the price **broke out upward**. You missed a BUY opportunity! Strong momentum often signals a breakout."
        warning = "**Pro tip:** When a stock sits quiet then moves up with **high volume**, that's a strong BUY signal."
    elif guess == "HOLD" and actual_dir == "DOWN":
        lesson = "You chose **HOLD** but the price **broke down**. Sometimes waiting costs money — exiting early protects your capital."
        warning = "**Pro tip:** A sharp drop with **high volume** is a warning sign — SELL or reduce your position."
    elif guess == "BUY" and actual_dir == "UP":
        lesson = "**Perfect BUY!** You correctly read the upward momentum. This is exactly how traders think — **ride the trend**."
        warning = "**Pro tip:** When a trade goes right, don't get greedy. Set a target and take profits."
    elif guess == "SELL" and actual_dir == "DOWN":
        lesson = "**Perfect SELL!** You caught the downtrend. Recognizing when sellers are in control is a skill many miss — you didn't."
        warning = "**Pro tip:** In a downtrend, patience pays. Don't rush to buy the \"bottom\"."
    else:
        lesson = "**Perfect HOLD!** You correctly identified the market was undecided. In choppy markets, doing nothing is often the smartest move."
        warning = "**Pro tip:** Not every day is a trading day. Sometimes the best trade is no trade."

    return {
        "trend_desc": trend_desc, "trend_color": trend_color,
        "expected_signal": expected_signal, "trend_pct": trend_pct,
        "lesson": lesson, "warning": warning,
        "actual_dir": actual_dir, "change_pct": current["change_pct"],
        "ticker": current["ticker"],
    }


def render_quiz_feedback(guess, current, was_correct):
    fb = generate_quiz_feedback(guess, current)

    if was_correct:
        reaction = random.choice(["🎉", "🔥", "💪", "⚡", "🏆"])
        st.markdown(f'<div class="ps-feedback correct"><span class="ps-reaction-emoji">{reaction}</span> CORRECT!</div>', unsafe_allow_html=True)
    else:
        reaction = random.choice(["😢", "😞", "💔", "😬"])
        st.markdown(f'<div class="ps-feedback wrong"><span class="ps-reaction-emoji">{reaction}</span> WRONG</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="ps-result-card">'
        f'<b>You guessed:</b> {guess} &nbsp;|&nbsp; '
        f'<b>Actual result:</b> Next close was {current["actual_next"]:.2f} '
        f'({fb["change_pct"]:+.2f}%) → <b>{fb["actual_dir"]}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="ps-teach-box">'
        f'<h4>📊 What the chart was actually telling you</h4>'
        f'<p>The last {CHART_DAYS} days of <b>{fb["ticker"]}</b> showed <b>{fb["trend_desc"]}</b> '
        f'(the price moved {fb["trend_pct"]:+.2f}% across the visible window). '
        f'Based on that pattern, the ideal signal was <b style="color:{fb["trend_color"]}">{fb["expected_signal"]}</b>.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    box_class = "ps-good-box" if was_correct else "ps-warn-box"
    st.markdown(
        f'<div class="{box_class}">'
        f'<b>📖 Lesson:</b> {fb["lesson"]}<br><br>{fb["warning"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🎓 How to read this chart — 60-second crash course"):
        st.markdown(
            """
            <div class="ps-lesson-card">
                <h4>🕯️ Reading a Candlestick</h4>
                <ul>
                    <li>🟢 <b>Green candle</b> = price closed HIGHER than it opened (buyers won)</li>
                    <li>🔴 <b>Red candle</b> = price closed LOWER than it opened (sellers won)</li>
                    <li>📍 <b>Top of the line</b> = highest price that day</li>
                    <li>📍 <b>Bottom of the line</b> = lowest price that day</li>
                    <li>➖ <b>Thick body</b> = big move, strong conviction</li>
                </ul>
            </div>
            <div class="ps-lesson-card">
                <h4>📈 Three Basic Patterns</h4>
                <ul>
                    <li><b>Uptrend</b> — higher highs + higher lows → BUY zone</li>
                    <li><b>Downtrend</b> — lower highs + lower lows → SELL zone</li>
                    <li><b>Sideways</b> — bouncing between a range → HOLD zone</li>
                </ul>
                <p>🎯 <b>Golden rule:</b> Trade WITH the trend, not against it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_floating_emoji(emoji):
    st.markdown(
        f'<div class="ps-emoji-container">'
        f'<div class="ps-floating-emoji" style="left:0;top:0;">{emoji}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_confetti():
    emojis = ["🎉", "✨", "🎊", "⭐", "💫", "🌟"]
    html = ""
    for i in range(15):
        left = random.randint(0, 100)
        delay = random.uniform(0, 1.5)
        emoji = random.choice(emojis)
        html += f'<div class="ps-confetti" style="left:{left}%;animation-delay:{delay}s;">{emoji}</div>'
    st.markdown(html, unsafe_allow_html=True)


def _handle_answer(correct, guess, user):
    """Process the user's answer, update stats, trigger emojis."""
    st.session_state["quiz_revealed"] = True
    st.session_state["quiz_last_correct"] = correct
    st.session_state["quiz_last_guess"] = guess
    st.session_state["quiz_round_results"] = st.session_state.get("quiz_round_results", []) + [correct]

    if correct:
        streak = st.session_state.get("quiz_streak", 0) + 1
        st.session_state["quiz_streak"] = streak
        if streak > st.session_state.get("quiz_best_streak", 0):
            st.session_state["quiz_best_streak"] = streak
        combo = min(max(streak, 1), 3)
        st.session_state["quiz_score"] += combo
        st.session_state["quiz_trigger_emoji"] = random.choice(["🎉", "🔥", "💪", "⚡"])
        if streak >= 3:
            st.session_state["quiz_show_confetti"] = True
    else:
        st.session_state["quiz_streak"] = 0
        st.session_state["quiz_lives"] = max(0, st.session_state.get("quiz_lives", MAX_LIVES) - 1)
        st.session_state["quiz_trigger_emoji"] = random.choice(["😢", "💔", "😞"])
        if st.session_state["quiz_lives"] <= 0:
            add_game_result(
                user,
                st.session_state["quiz_score"],
                ROUNDS_PER_GAME,
                st.session_state.get("quiz_best_streak", 0),
            )
            st.session_state["quiz_finished"] = True
    st.rerun()


def _reset_game():
    st.session_state["quiz_round"] = 1
    st.session_state["quiz_score"] = 0
    st.session_state["quiz_current"] = None
    st.session_state["quiz_revealed"] = False
    st.session_state["quiz_last_correct"] = None
    st.session_state["quiz_last_guess"] = None
    st.session_state["quiz_finished"] = False
    st.session_state["quiz_streak"] = 0
    st.session_state["quiz_best_streak"] = 0
    st.session_state["quiz_lives"] = MAX_LIVES
    st.session_state["quiz_hint_used"] = False
    st.session_state["quiz_hint_shown"] = False
    st.session_state["quiz_round_results"] = []
    st.session_state["quiz_trigger_emoji"] = None
    st.session_state["quiz_show_confetti"] = False


# ============================================================================
# HEADER + TICKER
# ============================================================================
now_str = datetime.datetime.now().strftime("%b %d, %Y — %H:%M")
st.markdown(
    f"""
    <div class="ps-header">
        <div>
            <h1>📈 PredictStock</h1>
            <p>AI-Powered Market Analysis, Multi-Currency Pricing &amp; Stock Quiz Game</p>
        </div>
        <div class="ps-badge">Live Data · {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_ticker_items = [
    ("NIFTY 50", "+0.62%", "up"), ("SENSEX", "+0.48%", "up"),
    ("NASDAQ", "-0.21%", "down"), ("S&P 500", "+0.15%", "up"),
    ("BTC/USD", "+1.84%", "up"), ("ETH/USD", "-0.52%", "down"),
    ("AAPL", "+0.94%", "up"), ("TSLA", "-1.12%", "down"),
    ("RELIANCE", "+0.33%", "up"), ("INFY", "+1.05%", "up"),
    ("GOLD", "+0.21%", "up"), ("USD/INR", "-0.08%", "down"),
]
_tape_html = "".join(f'<span>{n} <b class="{c}">{v}</b></span>' for n, v, c in _ticker_items)
st.markdown(
    f'<div class="ps-ticker-wrap"><div class="ps-ticker-track">{_tape_html}{_tape_html}</div></div>',
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("Market Controls")

if "ticker_symbol" not in st.session_state:
    st.session_state["ticker_symbol"] = "AAPL"


def _apply_search_pick():
    label = st.session_state.get("_search_pick_label")
    mapping = st.session_state.get("_search_label_map", {})
    if label in mapping:
        st.session_state["ticker_symbol"] = mapping[label]


search_query = st.sidebar.text_input(
    "🔍 Search any stock (name or symbol)", value="",
    placeholder="e.g. Amazon, Infosys, Tesla, Reliance",
)
search_results = search_stocks(search_query) if search_query.strip() else []

if search_results:
    match_labels = [label for _, label in search_results]
    st.session_state["_search_label_map"] = {label: sym for sym, label in search_results}
    st.sidebar.selectbox("Matching results — select one", match_labels, key="_search_pick_label", on_change=_apply_search_pick)

ticker_symbol = st.sidebar.text_input("Stock Ticker Symbol", key="ticker_symbol").upper().strip()
native_currency = get_native_currency(ticker_symbol)

display_currency = st.sidebar.selectbox(
    "Display Currency", CURRENCY_OPTIONS,
    index=CURRENCY_OPTIONS.index(native_currency) if native_currency in CURRENCY_OPTIONS else 0,
)

fx_rate, fx_ok = get_fx_rate(native_currency, display_currency)
if fx_ok:
    st.sidebar.caption(f"Native: **{native_currency}** → Display: **{display_currency}**")
elif native_currency != display_currency:
    st.sidebar.warning(f"⚠️ Couldn't fetch live rate. Showing in {native_currency}.")
    display_currency = native_currency
    fx_rate = 1.0

st.sidebar.header("Timeframe & Date Controls")
timeframe_option = st.sidebar.selectbox(
    "Select Historical Mode",
    ["Option 1: Current Year Data", "Option 2: Specific Selected Year", "Option 3: Custom Date Range"],
)
current_year = datetime.datetime.now().year

if timeframe_option == "Option 1: Current Year Data":
    start_date = f"{current_year}-01-01"
    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
elif timeframe_option == "Option 2: Specific Selected Year":
    selected_year = st.sidebar.number_input("Select Year", 2000, current_year, current_year - 1)
    start_date = f"{selected_year}-01-01"
    end_date = f"{selected_year}-12-31"
else:
    start_year = st.sidebar.number_input("Start Year", 2000, current_year, current_year - 2)
    start_date = st.sidebar.date_input("Start Date", datetime.date(int(start_year), 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.datetime.today().date())

if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.sidebar.error("Start date must be before end date.")
    st.stop()

st.sidebar.header("ML Model Parameters")
auto_tune = st.sidebar.checkbox("⚡ Auto-tune hyperparameters", value=False,
    help="Leave OFF on Streamlit Cloud free tier to avoid memory crashes.")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
test_size = st.sidebar.slider("Test Data Split Ratio", 0.1, 0.4, 0.2)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 20, 10)

# ============================================================================
# MAIN TABS
# ============================================================================
tab_analysis, tab_game = st.tabs(["📊 Analysis Dashboard", "🎮 Play & Learn"])

# ============================================================================
# TAB 1 — ANALYSIS
# ============================================================================
with tab_analysis:
    @st.cache_data(show_spinner=False)
    def load_data(ticker, start, end):
        df = yf.download(ticker, start=start, end=end, progress=False)
        return _flatten_columns(df)

    try:
        with st.spinner(f"Fetching live market data for {ticker_symbol}..."):
            df = load_data(ticker_symbol, start_date, end_date)
    except Exception as e:
        st.error(f"⚠️ Could not fetch data for '{ticker_symbol}'. Details: {e}")
        st.stop()

    if df.empty or "Close" not in df.columns:
        st.error(f"⚠️ No market data found for '{ticker_symbol}' in the selected range.")
        st.stop()

    if len(df) < 30:
        st.warning("⚠️ Very little historical data — indicators may be unreliable.")

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Bollinger_Mid"] = df["Close"].rolling(window=20).mean()
    df["Bollinger_Upper"] = df["Bollinger_Mid"] + (df["Close"].rolling(window=20).std() * 2)
    df["Bollinger_Lower"] = df["Bollinger_Mid"] - (df["Close"].rolling(window=20).std() * 2)
    df["Daily_Return_%"] = df["Close"].pct_change() * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

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

    latest_price = float(disp["Close"].iloc[-1])
    prev_price = float(disp["Close"].iloc[-2]) if len(disp) > 1 else latest_price
    price_change = latest_price - prev_price
    pct_change = (price_change / prev_price) * 100 if prev_price else 0.0

    lookback_252 = disp.tail(252)
    week52_high = float(lookback_252["High"].max())
    week52_low = float(lookback_252["Low"].min())

    st.markdown('<div class="ps-section-title">Market Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ps-sub">{ticker_symbol} · {start_date} → {end_date} · Priced in {display_currency}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Latest Close", fmt(latest_price, display_currency),
                f"{'▲' if price_change >= 0 else '▼'} {price_change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("Day High", fmt(float(disp['High'].iloc[-1]), display_currency))
    col3.metric("Day Low", fmt(float(disp['Low'].iloc[-1]), display_currency))
    col4.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
    col5.metric("52-Week Range", f"{fmt(week52_low, display_currency)} – {fmt(week52_high, display_currency)}")

    st.markdown('<div class="ps-section-title">📊 Price Action & Technical Indicators</div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{ticker_symbol} — Candlestick ({display_currency})", "Volume"),
    )
    fig.add_trace(go.Candlestick(x=disp.index, open=disp["Open"], high=disp["High"], low=disp["Low"], close=disp["Close"],
        name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp.index, y=disp["SMA_50"], line=dict(color="#facc15", width=1.4), name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp.index, y=disp["SMA_200"], line=dict(color="#38bdf8", width=1.4), name="SMA 200"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp.index, y=disp["EMA_20"], line=dict(color="#c084fc", width=1.2, dash="dot"), name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp.index, y=disp["Bollinger_Upper"], line=dict(color="#6b7280", width=1, dash="dash"), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp.index, y=disp["Bollinger_Lower"], line=dict(color="#6b7280", width=1, dash="dash"),
                              name="BB Lower", fill="tonexty", fillcolor="rgba(107,114,128,0.08)"), row=1, col=1)
    volume_colors = np.where(df["Close"] >= df["Open"], "#22c55e", "#ef4444")
    fig.add_trace(go.Bar(x=disp.index, y=df["Volume"], name="Volume", marker_color=volume_colors), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=650, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e6e9ef"), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
    fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ RSI (Relative Strength Index)"):
        latest_rsi = df["RSI_14"].dropna().iloc[-1] if df["RSI_14"].notna().any() else None
        if latest_rsi is not None:
            zone = "Overbought (>70)" if latest_rsi > 70 else "Oversold (<30)" if latest_rsi < 30 else "Neutral"
            st.write(f"Current RSI(14): **{latest_rsi:.1f}** — {zone}.")
        else:
            st.write("Not enough data points yet for RSI.")

    st.markdown('<div class="ps-section-title">🤖 AI Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-sub">Model: Random Forest · Features: OHLCV + SMA50 + EMA20 + Lags + Volatility + RSI + MACD</div>', unsafe_allow_html=True)

    if st.button("▶ Run AI Prediction"):
        feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_Signal",
                        "Lag_Close_1", "Lag_Close_2", "Lag_Close_3", "Volatility_20", "Return_Lag_1"]
        model_df = df[feature_cols].copy()
        model_df["Target"] = model_df["Close"].shift(-1)
        model_df.dropna(inplace=True)

        if len(model_df) > 60:
            X = model_df[feature_cols].values
            y = model_df["Target"].values
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if auto_tune and len(X_train) > 80:
                with st.spinner("Auto-tuning..."):
                    param_dist = {"n_estimators": [100, 150], "max_depth": [8, 12],
                                  "min_samples_split": [2, 5], "min_samples_leaf": [1, 2], "max_features": ["sqrt"]}
                    search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=1),
                        param_distributions=param_dist, n_iter=8, cv=TimeSeriesSplit(n_splits=3),
                        scoring="neg_mean_absolute_error", random_state=42, n_jobs=1)
                    search.fit(X_train, y_train)
                    best_params = search.best_params_
                    st.info(f"🎯 Best params: `{best_params}`")
            else:
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

            with st.spinner("Training model..."):
                model = RandomForestRegressor(random_state=42, n_jobs=1, **best_params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
            r2 = r2_score(y_test, predictions)
            direction_actual = np.sign(np.diff(y_test))
            direction_pred = np.sign(np.diff(predictions))
            dir_acc = float(np.mean(direction_actual == direction_pred) * 100) if len(direction_actual) > 0 else 0.0

            st.success("✅ Model trained successfully.")

            final_model = RandomForestRegressor(random_state=42, n_jobs=1, **best_params)
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
                st.markdown(f'<div style="padding-top:22px;">AI Signal:<br>'
                            f'<span class="ps-signal {css_class}">{arrow} {signal}</span></div>', unsafe_allow_html=True)
            st.caption(f"**95% CI:** {fmt(ci_low, display_currency)} — {fmt(ci_high, display_currency)} ({len(final_model.estimators_)} trees)")

            st.markdown("**Model Performance**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("MAE", fmt(mae * fx_rate, display_currency))
            m2.metric("RMSE", fmt(rmse * fx_rate, display_currency))
            m3.metric("R² Score", f"{r2:.3f}")
            m4.metric("Directional Acc.", f"{dir_acc:.1f}%")
            m5.metric("Test Samples", f"{len(y_test)}")

            importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            fi_fig = go.Figure(go.Bar(x=importances.values[::-1], y=importances.index[::-1], orientation="h",
                marker=dict(color=importances.values[::-1], colorscale=[[0, "#1e3a8a"], [0.5, "#3b82f6"], [1, "#a855f7"]])))
            fi_fig.update_layout(template="plotly_dark", height=380, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e6e9ef"), title="Feature Importance")
            st.plotly_chart(fi_fig, use_container_width=True)

            pred_dates = model_df.index[split_idx:]
            pred_df = pd.DataFrame({"Actual": y_test * fx_rate, "Predicted": predictions * fx_rate}, index=pred_dates)
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"], line=dict(color="#38bdf8", width=2), name="Actual"))
            pred_fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predicted"], line=dict(color="#facc15", width=2, dash="dot"), name="Predicted"))
            pred_fig.update_layout(template="plotly_dark", height=420, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e6e9ef"), hovermode="x unified", title=f"Actual vs Predicted ({display_currency})")
            st.plotly_chart(pred_fig, use_container_width=True)

            csv_data = pred_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
            st.download_button("⬇ Download Predictions (CSV)", data=csv_data,
                                file_name=f"{ticker_symbol}_predictions.csv", mime="text/csv")
        else:
            st.warning("Not enough data. Try a wider date range.")

    with st.expander("📄 View Raw Market Data"):
        display_table = disp[["Open", "High", "Low", "Close", "SMA_50", "SMA_200", "EMA_20", "RSI_14"]].copy()
        display_table["Volume"] = df["Volume"]
        st.dataframe(display_table.tail(50).round(2), use_container_width=True)

# ============================================================================
# TAB 2 — GAME
# ============================================================================
with tab_game:
    if "logged_in_user" not in st.session_state:
        st.session_state["logged_in_user"] = None

    # ---- LOGIN ----
    if st.session_state["logged_in_user"] is None:
        st.markdown('<div class="ps-section-title">🔐 Player Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-sub">Login to play the quiz and track your score.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            username = st.text_input("Username", key="_login_user")
            password = st.text_input("Password", type="password", key="_login_pass")
            if st.button("🎮 Log In", use_container_width=True):
                if USERS.get(username.strip().lower()) == password:
                    st.session_state["logged_in_user"] = username.strip().lower()
                    st.success(f"Welcome, **{username}**!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
        with col2:
            st.markdown("""
                **Demo accounts:**
                - `demo` / `demo123`
                - `admin` / `admin123`
                - `player1` / `pass1`
                - `player2` / `pass2`
                """)
    else:
        user = st.session_state["logged_in_user"]
        score_info = get_user_score(user)
        avatar_letter = user[0].upper() if user else "?"

        xp = score_info["score"]
        level = xp // 5 + 1
        xp_in_level = xp % 5
        xp_pct = (xp_in_level / 5) * 100

        st.markdown(f"""
            <div class="ps-player-card">
                <div class="ps-player-avatar">{avatar_letter}</div>
                <div style="flex:1;">
                    <div class="ps-player-name">⭐ Level {level} · {user}</div>
                    <div class="ps-player-meta">
                        🎮 Games: {score_info['games']} &nbsp;·&nbsp;
                        ✅ Correct: {score_info['correct']} / {score_info['total']} &nbsp;·&nbsp;
                        🎯 Accuracy: {(score_info['correct']/score_info['total']*100) if score_info['total'] else 0:.0f}% &nbsp;·&nbsp;
                        🔥 Best Streak: {score_info.get('best_streak', 0)}
                    </div>
                </div>
                <div class="ps-score-chip">🏆 {score_info['score']} pts</div>
            </div>
            <div class="ps-xp-label"><span>Level {level}</span><span>{xp_in_level} / 5 XP</span></div>
            <div class="ps-xp-wrap"><div class="ps-xp-fill" style="width:{xp_pct}%;"></div></div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state["logged_in_user"] = None
                st.rerun()

        game_tab1, game_tab2 = st.tabs(["🎯 Play Quiz", "📚 Training + Leaderboard"])

        # ---- QUIZ TAB ----
        with game_tab1:
            if "quiz_round" not in st.session_state:
                _reset_game()

            def _render_dots():
                dots = ""
                results = st.session_state.get("quiz_round_results", [])
                for i in range(ROUNDS_PER_GAME):
                    if i < len(results):
                        cls = "done" if results[i] else "wrong"
                    elif i == st.session_state["quiz_round"] - 1 and not st.session_state["quiz_finished"]:
                        cls = "active"
                    else:
                        cls = ""
                    dots += f'<div class="ps-round-dot {cls}"></div>'
                st.markdown(f'<div class="ps-round-dots">{dots}</div>', unsafe_allow_html=True)

            def _render_hud():
                streak = st.session_state.get("quiz_streak", 0)
                lives = st.session_state.get("quiz_lives", MAX_LIVES)
                combo = min(max(streak, 1), 3)
                hearts = "❤️" * lives + "🖤" * (MAX_LIVES - lives)
                streak_class = "active" if streak >= 2 else ""
                streak_text = f"🔥 {streak} in a row" if streak > 0 else "🔥 No streak"
                hint_text = "💡 Hint used" if st.session_state.get("quiz_hint_used", False) else "💡 Hint available"
                st.markdown(f'<div class="ps-hud-row">'
                            f'<div class="ps-hud-chip streak {streak_class}">{streak_text}</div>'
                            f'<div class="ps-hud-chip combo">⚡ {combo}x combo</div>'
                            f'<div class="ps-hud-chip lives">{hearts}</div>'
                            f'<div class="ps-hud-chip hint">{hint_text}</div>'
                            f'</div>', unsafe_allow_html=True)

            if st.session_state.get("quiz_trigger_emoji"):
                render_floating_emoji(st.session_state["quiz_trigger_emoji"])
                st.session_state["quiz_trigger_emoji"] = None
            if st.session_state.get("quiz_show_confetti"):
                render_confetti()
                st.session_state["quiz_show_confetti"] = False

            # ---- GAME OVER ----
            if st.session_state.get("quiz_finished"):
                _render_dots()
                final_score = st.session_state["quiz_score"]
                best_streak = st.session_state.get("quiz_best_streak", 0)

                if final_score == ROUNDS_PER_GAME:
                    st.markdown('<div class="ps-feedback correct">🏆 PERFECT GAME!</div>', unsafe_allow_html=True)
                    render_confetti()
                elif final_score >= ROUNDS_PER_GAME - 1:
                    st.markdown('<div class="ps-feedback correct">🎉 GREAT JOB!</div>', unsafe_allow_html=True)
                elif final_score >= ROUNDS_PER_GAME // 2:
                    st.markdown('<div class="ps-feedback correct">👍 NICE WORK!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ps-feedback wrong">💪 KEEP TRYING!</div>', unsafe_allow_html=True)

                st.markdown(f'<div style="text-align:center;font-size:32px;font-weight:900;color:#f8fafc;margin:20px 0;">{final_score} / {ROUNDS_PER_GAME}</div>', unsafe_allow_html=True)

                achievements = []
                if best_streak >= 3:
                    achievements.append("🔥 Streak Master")
                if final_score == ROUNDS_PER_GAME:
                    achievements.append("🏆 Perfectionist")
                if final_score >= 4:
                    achievements.append("🎯 Sharp Eye")
                if st.session_state.get("quiz_lives", 0) == MAX_LIVES:
                    achievements.append("❤️ Untouchable")
                if not st.session_state.get("quiz_hint_used", False):
                    achievements.append("🧠 No Hints Needed")

                if achievements:
                    st.markdown("### 🏅 Achievements Unlocked")
                    badges_html = "".join(f'<div class="ps-achievement">{a}</div>' for a in achievements)
                    st.markdown(f'<div class="ps-achievement-grid">{badges_html}</div>', unsafe_allow_html=True)

                if st.button("🔄 Play Again", use_container_width=True):
                    _reset_game()
                    st.rerun()

            # ---- WAITING FOR CHART ----
            elif st.session_state["quiz_current"] is None:
                _render_dots()
                _render_hud()
                if st.session_state.get("quiz_lives", 0) <= 1:
                    emoji, msg = "😬", "Careful — one life left!"
                elif st.session_state.get("quiz_streak", 0) >= 3:
                    emoji, msg = "🔥", "You're on fire!"
                elif st.session_state.get("quiz_streak", 0) >= 1:
                    emoji, msg = "😎", "Nice streak — keep going!"
                else:
                    emoji, msg = "🤔", "Ready for the next round?"

                st.markdown(f'<div style="text-align:center;font-size:24px;font-weight:800;color:#f8fafc;margin:16px 0;">'
                            f'<span class="ps-reaction-emoji">{emoji}</span> {msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center;color:#94a3b8;margin-bottom:16px;">'
                            f'Round {st.session_state["quiz_round"]} of {ROUNDS_PER_GAME} — '
                            f'look at the chart → guess BUY / SELL / HOLD</div>', unsafe_allow_html=True)
                if st.button("▶ Show Chart", use_container_width=True):
                    with st.spinner("Loading chart..."):
                        st.session_state["quiz_current"] = fetch_quiz_round()
                        st.session_state["quiz_revealed"] = False
                        st.session_state["quiz_last_correct"] = None
                        st.session_state["quiz_last_guess"] = None
                        st.session_state["quiz_hint_shown"] = False
                    st.rerun()

            # ---- CURRENT ROUND ----
            else:
                current = st.session_state["quiz_current"]
                if current is None:
                    st.error("Couldn't load chart. Try again.")
                    if st.button("Retry"):
                        st.session_state["quiz_current"] = None
                        st.rerun()
                else:
                    _render_dots()
                    _render_hud()
                    st.markdown(f'<div style="text-align:center;font-size:22px;font-weight:800;color:#f8fafc;">'
                                f'Round {st.session_state["quiz_round"]} / {ROUNDS_PER_GAME} '
                                f'&nbsp;·&nbsp; Score: {st.session_state["quiz_score"]}</div>', unsafe_allow_html=True)

                    quiz_fig = go.Figure(go.Candlestick(x=current["df"].index,
                        open=current["df"]["Open"], high=current["df"]["High"],
                        low=current["df"]["Low"], close=current["df"]["Close"],
                        increasing_line_color="#22c55e", decreasing_line_color="#ef4444", name="Price"))
                    quiz_fig.update_layout(template="plotly_dark", height=400,
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e6e9ef"),
                        title=dict(text=f"{current['ticker']} — Last {CHART_DAYS} Days", font=dict(size=16)),
                        xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10))
                    quiz_fig.update_xaxes(showgrid=True, gridcolor="#1f2530")
                    quiz_fig.update_yaxes(showgrid=True, gridcolor="#1f2530")
                    st.plotly_chart(quiz_fig, use_container_width=True)

                    if not st.session_state["quiz_revealed"]:
                        if st.session_state.get("quiz_hint_shown", False):
                            hint = generate_hint(current)
                            st.markdown(f'<div class="ps-hint-box"><b>💡 Hint:</b> {hint}</div>', unsafe_allow_html=True)

                        st.markdown('<div style="text-align:center;margin:14px 0 4px 0;">'
                                    '<span class="ps-reaction-emoji">🤔</span></div>', unsafe_allow_html=True)
                        st.markdown('<div style="text-align:center;color:#cbd5e1;font-size:15px;'
                                    'font-weight:600;">🎯 What will happen next?</div>', unsafe_allow_html=True)

                        if not st.session_state.get("quiz_hint_used", False):
                            if st.button("💡 Use Hint (costs 1 point)"):
                                st.session_state["quiz_hint_used"] = True
                                st.session_state["quiz_hint_shown"] = True
                                st.session_state["quiz_score"] = max(0, st.session_state["quiz_score"] - 1)
                                st.rerun()
                        elif not st.session_state.get("quiz_hint_shown", False):
                            st.caption("💡 Hint already used this game.")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("📈 BUY", use_container_width=True, key="ans_buy"):
                                correct = score_guess("BUY", current["actual_direction"])
                                _handle_answer(correct, "BUY", user)
                        with c2:
                            if st.button("📉 SELL", use_container_width=True, key="ans_sell"):
                                correct = score_guess("SELL", current["actual_direction"])
                                _handle_answer(correct, "SELL", user)
                        with c3:
                            if st.button("✋ HOLD", use_container_width=True, key="ans_hold"):
                                correct = score_guess("HOLD", current["actual_direction"])
                                _handle_answer(correct, "HOLD", user)
                    else:
                        render_quiz_feedback(
                            st.session_state["quiz_last_guess"] or "BUY",
                            current,
                            st.session_state["quiz_last_correct"],
                        )
                        if st.button("➡ Next Round", use_container_width=True):
                            st.session_state["quiz_revealed"] = False
                            st.session_state["quiz_last_correct"] = None
                            st.session_state["quiz_last_guess"] = None
                            st.session_state["quiz_hint_shown"] = False
                            if st.session_state["quiz_round"] >= ROUNDS_PER_GAME:
                                add_game_result(user, st.session_state["quiz_score"], ROUNDS_PER_GAME,
                                                st.session_state.get("quiz_best_streak", 0))
                                st.session_state["quiz_finished"] = True
                                st.session_state["quiz_current"] = None
                            else:
                                st.session_state["quiz_round"] += 1
                                st.session_state["quiz_current"] = None
                            st.rerun()

        # ---- TRAINING + LEADERBOARD TAB ----
        with game_tab2:
            st.markdown('<div class="ps-section-title">📚 Stock Training Session</div>', unsafe_allow_html=True)
            st.markdown('<div class="ps-sub">Learn the basics before you play.</div>', unsafe_allow_html=True)

            with st.expander("📖 Lesson 1 — What is a Candlestick?"):
                st.markdown("""
                    <div class="ps-lesson-card">
                        <h4>🕯️ Anatomy of a Candle</h4>
                        <p>Each candle = <b>one time period</b> (usually 1 day). It packs 4 numbers:</p>
                        <ul>
                            <li><b>Open</b> — price when the market opened</li>
                            <li><b>High</b> — highest price touched</li>
                            <li><b>Low</b> — lowest price touched</li>
                            <li><b>Close</b> — price when market closed</li>
                        </ul>
                        <p>🟢 <b>Green (bullish)</b> = close above open → buyers won</p>
                        <p>🔴 <b>Red (bearish)</b> = close below open → sellers won</p>
                        <p>➖ <b>Thick body</b> = big move. <b>Thin body (doji)</b> = indecision.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📖 Lesson 2 — What is Market Cap?"):
                st.markdown("""
                    <div class="ps-lesson-card">
                        <h4>💰 Total Value of the Company</h4>
                        <p><b>Market Cap = Share Price × Total Shares Outstanding</b></p>
                        <p>Think of it as the price tag on the entire business.</p>
                        <ul>
                            <li>🐘 <b>Large Cap (&gt;$10B)</b> — Apple, Microsoft, Reliance. Stable, slower.</li>
                            <li>🐕 <b>Mid Cap ($2B–$10B)</b> — growing companies, balanced risk.</li>
                            <li>🐁 <b>Small Cap (&lt;$2B)</b> — young, risky, big growth potential.</li>
                        </ul>
                        <p>💡 <b>Why it matters:</b> A ₹10 stock in a ₹1 lakh company is very different from a ₹10 stock in a ₹10,000 crore company.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📖 Lesson 3 — BUY / SELL / HOLD Signals"):
                st.markdown("""
                    <div class="ps-lesson-card">
                        <h4>🎯 When to Choose What</h4>
                        <ul>
                            <li><b>BUY</b> → you expect the price to rise &gt;+1%</li>
                            <li><b>SELL</b> → you expect the price to fall &gt;-1%</li>
                            <li><b>HOLD</b> → you expect it to stay flat (±1%)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📖 Lesson 4 — Reading Trends"):
                st.markdown("""
                    <div class="ps-lesson-card">
                        <h4>📈 Three Patterns You Must Know</h4>
                        <ul>
                            <li><b>Uptrend</b> — higher highs + higher lows → BUY zone.</li>
                            <li><b>Downtrend</b> — lower highs + lower lows → SELL zone.</li>
                            <li><b>Sideways</b> — price bounces in a range → HOLD zone.</li>
                        </ul>
                        <p>🎯 <b>Golden rule:</b> "The trend is your friend."</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🏆 Leaderboard")
            board = get_leaderboard()
            if not board:
                st.info("No scores yet. Be the first to play!")
            else:
                for i, (u, s) in enumerate(board, 1):
                    row_cls = "gold" if i == 1 else "silver" if i == 2 else "bronze" if i == 3 else ""
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                    st.markdown(f'<div class="ps-lb-row {row_cls}">'
                                f'<span class="ps-lb-name">{medal} {u}</span>'
                                f'<span class="ps-lb-score">{s} pts</span></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Your Stats")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Games", score_info["games"])
            s2.metric("Correct", score_info["correct"])
            s3.metric("Total Q", score_info["total"])
            s4.metric("Score", score_info["score"])
            s5.metric("Best Streak", score_info.get("best_streak", 0))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    """
    <div class="ps-footer">
        PredictStock Dashboard · Data via Yahoo Finance (yfinance) · FX rates via live pairs<br>
        Educational/research prototype — not financial advice. No guaranteed accuracy.
        <div class="ps-footer-badges">
            <span>Python</span><span>Streamlit</span><span>Plotly</span>
            <span>scikit-learn</span><span>yfinance</span><span>Random Forest</span>
            <span>Quiz Game</span><span>Streaks</span><span>Combos</span>
            <span>Achievements</span><span>Animated Emojis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
