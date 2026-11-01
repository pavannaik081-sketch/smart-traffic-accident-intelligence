import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Traffic & Accident Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM DASHBOARD STYLING
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
    color: #1e293b;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4 {
    color: #172554 !important;
    opacity: 1 !important;
}
.stApp p,
.stApp li,
.stApp [data-testid="stMarkdownContainer"] {
    color: #334155;
}
.dashboard-subtitle {
    color: #475569 !important;
    font-size: 15px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #dce5f1;
    border-radius: 14px;
    padding: 22px 18px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}
.kpi-label {
    color: #475569 !important;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}
.kpi-value {
    color: #172554 !important;
    font-size: 29px;
    font-weight: 800;
    line-height: 1.2;
}
.kpi-description {
    color: #64748b !important;
    font-size: 12px;
    margin-top: 8px;
}
section[data-testid="stSidebar"] {
    background-color: #eef3fa;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #1e293b !important;
}
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp label {
    color: #263653 !important;
    font-weight: 600;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #52627a !important;
}
.stApp input,
.stApp textarea,
