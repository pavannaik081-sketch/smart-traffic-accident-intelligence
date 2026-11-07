import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 1. PAGE CONFIGURATION & METADATA
# =========================================================
st.set_page_config(
    page_title="NYC Traffic & Accident Intelligence Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================================================
# 2. DOMAINS DEFINITION & STATE SYNCHRONIZATION
# =========================================================
DOMAINS = [
    "Executive Overview & KPIs",
    "Borough & Spatial Domain",
    "Temporal & Time Dynamics",
    "Severity & Casualties",
    "Vehicles & Contributing Factors",
    "Geospatial Maps & Hotspots",
    "ML Severity Predictor & Evaluation",
    "Accident Data Explorer"
]

DOMAIN_SHORT_NAMES = {
    "Executive Overview & KPIs": "Overview",
    "Borough & Spatial Domain": "Boroughs",
    "Temporal & Time Dynamics": "Temporal",
    "Severity & Casualties": "Severity",
    "Vehicles & Contributing Factors": "Vehicles",
    "Geospatial Maps & Hotspots": "Maps & Hotspots",
    "ML Severity Predictor & Evaluation": "ML Predictor",
    "Accident Data Explorer": "Explorer"
}

# State management callbacks to eliminate StreamlitWidgetAlreadyInstantiatedError
if "active_domain" not in st.session_state:
    st.session_state["active_domain"] = DOMAINS[0]

if "sidebar_domain_selector" not in st.session_state:
    st.session_state["sidebar_domain_selector"] = st.session_state["active_domain"]

def on_sidebar_radio_change():
    st.session_state["active_domain"] = st.session_state["sidebar_domain_selector"]

def set_active_domain(domain_name):
    st.session_state["active_domain"] = domain_name
    st.session_state["sidebar_domain_selector"] = domain_name

# Sync selector before widget creation
st.session_state["sidebar_domain_selector"] = st.session_state["active_domain"]

# =========================================================
# 3. THEME & APPEARANCE SYSTEM
# =========================================================
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>
    <span style="font-size: 15px; font-weight: 700; letter-spacing: 0.5px;">Dashboard Appearance</span>
</div>
""", unsafe_allow_html=True)

theme_choice = st.sidebar.radio(
    "Theme Mode",
    ["Dark Mode", "Light Mode"],
    horizontal=True,
    label_visibility="collapsed",
    key="app_theme_mode"
)

DARK_MODE = theme_choice == "Dark Mode"

if DARK_MODE:
    BG_COLOR = "#0b0f19"
    SURFACE_COLOR = "#131b2e"
    SURFACE_CARD = "#192237"
    SIDEBAR_BG = "#0d1424"
    TEXT_MAIN = "#f8fafc"
    TEXT_MUTED = "#cbd5e1"
    BORDER_COLOR = "#23304a"
    INPUT_BG = "#162032"
    INPUT_TEXT = "#f8fafc"
    INPUT_BORDER = "#3b4f6f"
    ACCENT_BLUE = "#38bdf8"
    ACCENT_PURPLE = "#a855f7"
    TAB_INACTIVE = "#e2e8f0"
    COLOR_FATAL = "#ef4444"
    COLOR_INJURY = "#f59e0b"
    COLOR_DAMAGE = "#38bdf8"
    GRID_COLOR = "#1e293b"
    BUTTON_GRADIENT = "linear-gradient(135deg, #2563eb, #7c3aed)"
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    BG_COLOR = "#f8fafc"
    SURFACE_COLOR = "#ffffff"
    SURFACE_CARD = "#f1f5f9"
    SIDEBAR_BG = "#ffffff"
    TEXT_MAIN = "#0f172a"
    TEXT_MUTED = "#334155"
    BORDER_COLOR = "#cbd5e1"
    INPUT_BG = "#ffffff"
    INPUT_TEXT = "#0f172a"
    INPUT_BORDER = "#94a3b8"
    ACCENT_BLUE = "#0284c7"
    ACCENT_PURPLE = "#9333ea"
    TAB_INACTIVE = "#334155"
    COLOR_FATAL = "#dc2626"
    COLOR_INJURY = "#d97706"
    COLOR_DAMAGE = "#0284c7"
    GRID_COLOR = "#e2e8f0"
    BUTTON_GRADIENT = "linear-gradient(135deg, #1d4ed8, #2563eb)"
    PLOTLY_TEMPLATE = "plotly_white"

SEVERITY_COLORS = {
    "Property Damage": COLOR_DAMAGE,
    "Injury": COLOR_INJURY,
    "Fatal": COLOR_FATAL
}

BOROUGH_COLORS = {
    "BROOKLYN": "#38bdf8" if DARK_MODE else "#0284c7",
    "QUEENS": "#34d399" if DARK_MODE else "#059669",
    "MANHATTAN": "#a78bfa" if DARK_MODE else "#7c3aed",
    "BRONX": "#fb923c" if DARK_MODE else "#ea580c",
    "STATEN ISLAND": "#f472b6" if DARK_MODE else "#db2777",
    "Unknown": "#94a3b8" if DARK_MODE else "#64748b"
}

# Responsive Modern CSS with High-Contrast Light & Dark Mode
st.markdown(f"""
<style>
    /* Main app body */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_MAIN};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: {BG_COLOR};
    }}
    
    [data-testid="stHeader"] {{
        background-color: {BG_COLOR};
    }}
    
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5%;
        padding-right: 2.5%;
        max-width: 1680px;
    }}
    
    /* Typography contrast */
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_MAIN} !important;
        font-weight: 700;
    }}
    
    p, span, label, div {{
        color: inherit;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
        border-right: 1px solid {BORDER_COLOR};
    }}
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {{
        color: {TEXT_MAIN} !important;
    }}
    
    /* Input Elements, Selectboxes, Multiselects */
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"],
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {{
        background-color: {INPUT_BG} !important;
        color: {INPUT_TEXT} !important;
        border: 1.5px solid {INPUT_BORDER} !important;
        border-radius: 8px !important;
    }}
    
    /* Multiselect text & tags */
    .stMultiSelect div[data-baseweb="select"] span {{
        color: {INPUT_TEXT} !important;
    }}
    
    .stSelectbox div[data-baseweb="select"] div,
    .stMultiSelect div[data-baseweb="select"] div {{
        color: {INPUT_TEXT} !important;
    }}
    
    /* Dropdown Menus and Popovers */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {{
        background-color: {SURFACE_COLOR} !important;
        color: {TEXT_MAIN} !important;
    }}
    
    li[role="option"]:hover {{
        background-color: {SURFACE_CARD} !important;
    }}
    
    li[role="option"] * {{
        color: {TEXT_MAIN} !important;
    }}

    /* Metrics Cards */
    [data-testid="stMetric"] {{
        background-color: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }}
    
    [data-testid="stMetricLabel"] p {{
        font-size: 13px !important;
        font-weight: 600 !important;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 26px !important;
        font-weight: 700 !important;
        color: {TEXT_MAIN} !important;
    }}
    
    /* Tabs Navigation - High Contrast in Both Modes */
    button[data-baseweb="tab"] {{
        color: {TAB_INACTIVE} !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        background: transparent !important;
        padding: 10px 18px !important;
        opacity: 0.9 !important;
    }}
    
    button[data-baseweb="tab"] * {{
        color: inherit !important;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT_BLUE} !important;
        border-bottom: 3px solid {ACCENT_BLUE} !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    
    button[data-baseweb="tab"]:hover {{
        color: {TEXT_MAIN} !important;
        opacity: 1 !important;
    }}

    /* Primary buttons (active domain & submit) */
    button[kind="primary"],
    [data-testid="stFormSubmitButton"] button {{
        background: {BUTTON_GRADIENT} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.1rem !important;
        min-height: 42px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    
    button[kind="primary"] *,
    [data-testid="stFormSubmitButton"] button * {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button:hover {{
        opacity: 0.92 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    }}

    /* Secondary buttons (inactive domain pills) */
    button[kind="secondary"] {{
        background-color: {SURFACE_CARD} !important;
        color: {TEXT_MAIN} !important;
        border: 1.5px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
        min-height: 42px !important;
        transition: all 0.2s ease !important;
    }}
    
    button[kind="secondary"] * {{
        color: {TEXT_MAIN} !important;
        font-weight: 600 !important;
    }}
    
    button[kind="secondary"]:hover {{
        border-color: {ACCENT_BLUE} !important;
        color: {ACCENT_BLUE} !important;
        transform: translateY(-1px);
    }}
    
    button[kind="secondary"]:hover * {{
        color: {ACCENT_BLUE} !important;
    }}

    /* Plotly Chart Container */
    [data-testid="stPlotlyChart"] {{
        background-color: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 8px;
    }}
    
    /* Alerts and Warnings */
    [data-testid="stAlert"] * {{
        color: {TEXT_MAIN} !important;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. CHART STYLING HELPER
# =========================================================
def format_fig(fig, title="", height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial", size=12, color=TEXT_MAIN),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=16, color=TEXT_MAIN),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        height=height,
        margin=dict(l=35, r=25, t=55 if title else 25, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MAIN, size=11)
        ),
        xaxis=dict(
            title_font=dict(color=TEXT_MAIN, size=12),
            tickfont=dict(color=TEXT_MUTED, size=11),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR
        ),
        yaxis=dict(
            title_font=dict(color=TEXT_MAIN, size=12),
            tickfont=dict(color=TEXT_MUTED, size=11),
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR
        )
    )
    return fig

# =========================================================
# 5. ROBUST DATA LOADING (MULTI-TIER CACHED)
# =========================================================
@st.cache_data(show_spinner="Loading accident intelligence dataset...")
def get_dataset():
    candidates = [
        PROJECT_ROOT / "dataset" / "processed" / "traffic_accidents_dashboard.csv",
        PROJECT_ROOT / "dataset" / "processed" / "traffic_accidents_cleaned.csv",
        PROJECT_ROOT / "dataset" / "raw" / "traffic_accidents.csv"
    ]
    
    source_path = None
    for cand in candidates:
        if cand.exists():
            source_path = cand
            break
            
    huggingface_url = (
        "https://huggingface.co/datasets/pavan-data-lab/nyc-traffic-accidents/resolve/main/"
        "traffic_accidents_dashboard.csv"
    )
    
    file_to_read = str(source_path) if source_path else huggingface_url
    
    required_cols = {
        "crash_date", "crash_time", "borough", "latitude", "longitude",
        "number_of_persons_injured", "number_of_persons_killed",
        "contributing_factor_vehicle_1", "vehicle_type_code_1"
    }
    
    def norm(col):
        return str(col).strip().lower().replace(" ", "_")
        
    # Cloud-safe row limit to prevent exceeding container memory limits
    max_cloud_rows = 600_000 if not source_path else None

    df = pd.read_csv(
        file_to_read,
        nrows=max_cloud_rows,
        low_memory=False,
        usecols=lambda c: norm(c) in required_cols
    )
    
    df.columns = [norm(c) for c in df.columns]
    
    # Process Datetime
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df.dropna(subset=["crash_date"]).copy()
    
    # Hour extraction
    time_series = df["crash_time"].fillna("00:00").astype(str)
    df["crash_hour"] = pd.to_numeric(time_series.str.slice(0, 2), errors="coerce").fillna(0).astype("int8")
    
    # Coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float32")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float32")
    
    # Casualties
    df["injured"] = pd.to_numeric(df["number_of_persons_injured"], errors="coerce").fillna(0).astype("int16")
    df["killed"] = pd.to_numeric(df["number_of_persons_killed"], errors="coerce").fillna(0).astype("int16")
    
    # Categoricals with clean string sanitization
    df["borough"] = df["borough"].fillna("Unknown").astype(str).str.strip().str.upper()
    df["borough"] = df["borough"].replace({"": "Unknown", "NAN": "Unknown"}).astype("category")
    
    df["factor"] = df["contributing_factor_vehicle_1"].fillna("Unspecified").astype(str).str.strip()
    df["factor"] = df["factor"].replace({"": "Unspecified", "NAN": "Unspecified", "1": "Unspecified"}).astype("category")
    
    df["vehicle"] = df["vehicle_type_code_1"].fillna("Unknown").astype(str).str.strip()
    df["vehicle"] = df["vehicle"].replace({"": "Unknown", "NAN": "Unknown"}).astype("category")
    
    # Drop redundant raw columns to conserve memory
    df.drop(columns=[
        "crash_time", "number_of_persons_injured", "number_of_persons_killed",
        "contributing_factor_vehicle_1", "vehicle_type_code_1"
    ], inplace=True, errors="ignore")

    # Severity calculation
    conditions = [df["killed"] > 0, df["injured"] > 0]
    choices = ["Fatal", "Injury"]
    df["severity"] = pd.Categorical(np.select(conditions, choices, default="Property Damage"))
    
    # Temporal attributes
    df["year"] = df["crash_date"].dt.year.astype("int16")
    df["month"] = df["crash_date"].dt.month.astype("int8")
    df["month_name"] = df["crash_date"].dt.strftime("%b").fillna("Unknown").astype("category")
    df["day_of_week"] = df["crash_date"].dt.day_name().fillna("Unknown").astype("category")
    
    import gc
    gc.collect()

    return df, file_to_read

@st.cache_data
def load_html_content(file_path_str: str) -> str:
    with open(file_path_str, "r", encoding="utf-8") as f:
        return f.read()

# Load data safely
try:
    df, active_source = get_dataset()
except Exception as err:
    st.error(f"Failed to load dataset: {err}")
    st.info("Please verify that the dataset file exists in dataset/processed/ or dataset/raw/")
    st.stop()

# =========================================================
# 6. SIDEBAR CONTROLS & DOMAIN SELECTOR
# =========================================================
# Professional Logo in Sidebar
logo_fill = "#38bdf8" if DARK_MODE else "#0284c7"
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid {BORDER_COLOR};">
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="36" height="36" rx="8" fill="{logo_fill}" fill-opacity="0.15"/>
        <path d="M18 6V30M6 18H30" stroke="{logo_fill}" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="18" cy="18" r="4" fill="{logo_fill}"/>
        <circle cx="9" cy="9" r="2.5" fill="#ef4444"/>
        <circle cx="27" cy="9" r="2.5" fill="#f59e0b"/>
        <circle cx="27" cy="27" r="2.5" fill="#10b981"/>
    </svg>
    <div>
        <div style="font-weight: 800; font-size: 15px; letter-spacing: 0.5px; color: {TEXT_MAIN}; line-height: 1.2;">NYC TRAFFIC</div>
        <div style="font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; letter-spacing: 0.8px;">INTELLIGENCE SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="3 11 22 2 13 21 11 13 3 11"></polygon>
    </svg>
    <span style="font-size: 14px; font-weight: 700;">Analysis Domain</span>
</div>
""", unsafe_allow_html=True)

current_domain_idx = DOMAINS.index(st.session_state["active_domain"]) if st.session_state["active_domain"] in DOMAINS else 0

selected_sidebar = st.sidebar.radio(
    "Select Domain to Explore:",
    DOMAINS,
    index=current_domain_idx,
    key="sidebar_domain_selector",
    on_change=on_sidebar_radio_change,
    label_visibility="collapsed"
)

# Active domain resolution
selected_domain = st.session_state["active_domain"]

st.sidebar.markdown(f"<hr style='margin: 15px 0; border-color: {BORDER_COLOR};'>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
        <circle cx="12" cy="10" r="3"></circle>
    </svg>
    <span style="font-size: 14px; font-weight: 700;">Borough Domain Focus</span>
</div>
""", unsafe_allow_html=True)

available_boroughs = sorted([str(b) for b in df["borough"].dropna().unique() if str(b).strip()])
borough_options_list = ["All NYC Boroughs", "Custom Multi-Borough"] + available_boroughs

borough_choice = st.sidebar.selectbox(
    "Choose Borough Domain:",
    options=borough_options_list,
    index=0
)

if borough_choice == "All NYC Boroughs":
    selected_boroughs = available_boroughs
elif borough_choice == "Custom Multi-Borough":
    selected_boroughs = st.sidebar.multiselect(
        "Select Boroughs:",
        options=available_boroughs,
        default=available_boroughs
    )
    if not selected_boroughs:
        selected_boroughs = available_boroughs
else:
    selected_boroughs = [borough_choice]

# Severity selection
severity_options = ["Property Damage", "Injury", "Fatal"]
selected_severities = st.sidebar.multiselect(
    "Accident Severity",
    options=severity_options,
    default=severity_options
)

if not selected_severities:
    selected_severities = severity_options

# Date range selector
min_dt = df["crash_date"].min().date()
max_dt = df["crash_date"].max().date()

date_range = st.sidebar.date_input(
    "Crash Date Range",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt
)

if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
    elif len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = min_dt, max_dt
else:
    start_date = end_date = date_range

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

# Apply filters
mask = (
    (df["crash_date"] >= start_ts) &
    (df["crash_date"] < end_ts) &
    (df["borough"].isin(selected_boroughs)) &
    (df["severity"].isin(selected_severities))
)

filtered_df = df.loc[mask]

# Sidebar Status
st.sidebar.markdown(f"<hr style='margin: 15px 0; border-color: {BORDER_COLOR};'>", unsafe_allow_html=True)
st.sidebar.caption(f"**Records Loaded:** {len(df):,}")
st.sidebar.caption(f"**Filtered Records:** {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}%)")
if st.sidebar.button("Reset All Filters", use_container_width=True):
    st.session_state["active_domain"] = DOMAINS[0]
    st.session_state["sidebar_domain_selector"] = DOMAINS[0]
    st.rerun()

# =========================================================
# 7. HEADER & TOP DOMAIN NAVIGATION PILLS
# =========================================================
header_logo_color = "#38bdf8" if DARK_MODE else "#0284c7"

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 12px;">
    <svg width="42" height="42" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="42" height="42" rx="10" fill="{header_logo_color}" fill-opacity="0.12"/>
        <rect x="1" y="1" width="40" height="40" rx="9" stroke="{header_logo_color}" stroke-opacity="0.3" stroke-width="1.5"/>
        <path d="M21 7V35M7 21H35" stroke="{header_logo_color}" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="21" cy="21" r="5" fill="{header_logo_color}"/>
        <circle cx="11" cy="11" r="3" fill="#ef4444"/>
        <circle cx="31" cy="11" r="3" fill="#f59e0b"/>
        <circle cx="31" cy="31" r="3" fill="#10b981"/>
    </svg>
    <div>
        <h1 style="font-size: 26px; font-weight: 800; margin: 0; line-height: 1.2; color: {TEXT_MAIN};">
            NYC Smart Traffic & Accident Intelligence Platform
        </h1>
        <p style="color: {TEXT_MUTED}; font-size: 14px; margin: 2px 0 0 0;">
            Multi-Domain Urban Safety Analytics & Machine Learning Decision Support System
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Domain Quick-Pills / Navigation Buttons with on_click callback
cols = st.columns(len(DOMAINS))
for idx, (col, dom) in enumerate(zip(cols, DOMAINS)):
    is_active = (selected_domain == dom)
    short_lbl = DOMAIN_SHORT_NAMES.get(dom, dom)
    col.button(
        short_lbl,
        key=f"top_nav_btn_{idx}",
        type="primary" if is_active else "secondary",
        on_click=set_active_domain,
        args=(dom,),
        use_container_width=True
    )

st.markdown(f"<hr style='margin-top: 10px; margin-bottom: 20px; border-color: {BORDER_COLOR};'>", unsafe_allow_html=True)

# Defensive check for empty filtered dataframe
if filtered_df.empty:
    st.warning("No accident records match the selected filter criteria. Please adjust your date range, borough, or severity filters.")
    st.info("Click 'Reset All Filters' in the sidebar to return to the full dataset.")
    st.stop()

# Key aggregate metrics
total_accidents = len(filtered_df)
total_injuries = int(filtered_df["injured"].sum())
total_fatalities = int(filtered_df["killed"].sum())
total_casualties = total_injuries + total_fatalities
fatal_record_count = int((filtered_df["severity"] == "Fatal").sum())
injury_record_count = int((filtered_df["severity"] == "Injury").sum())
property_damage_count = int((filtered_df["severity"] == "Property Damage").sum())

fatality_rate = (total_fatalities / total_accidents * 100) if total_accidents > 0 else 0
injury_rate = (total_injuries / total_accidents * 100) if total_accidents > 0 else 0

# =========================================================
# DOMAIN 1: EXECUTIVE OVERVIEW & KPIS
# =========================================================
if selected_domain == "Executive Overview & KPIs":
    st.subheader("Executive Overview & Key Safety Indicators")
    
    # 4 Hero KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Total Collisions",
        f"{total_accidents:,}",
        help="Total traffic accident incidents recorded in the selected domain and filter scope"
    )
    k2.metric(
        "Total Persons Injured",
        f"{total_injuries:,}",
        delta=f"{injury_rate:.1f} per 100 crashes",
        delta_color="inverse",
        help="Number of people injured across all selected collisions"
    )
    k3.metric(
        "Total Fatalities",
        f"{total_fatalities:,}",
        delta=f"{fatality_rate:.2f}% fatality rate",
        delta_color="inverse",
        help="Number of individuals who tragically lost their lives"
    )
    k4.metric(
        "Casualties / Crash",
        f"{(total_casualties / total_accidents):.2f}" if total_accidents > 0 else "0",
        help="Average number of injured or killed individuals per accident"
    )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Row 2: Severity Distribution + Borough Breakdown
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Collision Severity Proportion")
        sev_counts = filtered_df["severity"].value_counts().reindex(severity_options).fillna(0).reset_index()
        sev_counts.columns = ["Severity", "Count"]
        
        fig_sev = px.pie(
            sev_counts,
            names="Severity",
            values="Count",
            hole=0.55,
            color="Severity",
            color_discrete_map=SEVERITY_COLORS
        )
        fig_sev.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color=SURFACE_COLOR, width=2))
        )
        st.plotly_chart(format_fig(fig_sev, "Accident Severity Distribution", height=380), use_container_width=True)

    with c2:
        st.markdown("#### Collisions by Borough Domain")
        b_counts = filtered_df["borough"].value_counts().reset_index()
        b_counts.columns = ["Borough", "Accidents"]
        
        fig_borough = px.bar(
            b_counts,
            x="Accidents",
            y="Borough",
            orientation="h",
            color="Borough",
            color_discrete_map=BOROUGH_COLORS,
            text_auto=True
        )
        fig_borough.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
        st.plotly_chart(format_fig(fig_borough, "Accident Volume by Borough", height=380), use_container_width=True)

    # Row 3: Hourly Crash Trend Ribbon
    st.markdown("#### Hourly Collision Activity Curve")
    hourly_trend = filtered_df.groupby("crash_hour").size().reindex(range(24), fill_value=0).reset_index(name="Accidents")
    
    fig_hourly_curve = px.area(
        hourly_trend,
        x="crash_hour",
        y="Accidents",
        markers=True,
        color_discrete_sequence=[ACCENT_BLUE]
    )
    fig_hourly_curve.update_xaxes(
        tickmode="linear",
        dtick=2,
        title="Hour of the Day (0:00 - 23:00)"
    )
    fig_hourly_curve.update_yaxes(title="Accident Volume")
    st.plotly_chart(format_fig(fig_hourly_curve, "24-Hour Traffic Accident Profile", height=320), use_container_width=True)

# =========================================================
# DOMAIN 2: BOROUGH & SPATIAL DOMAIN
# =========================================================
elif selected_domain == "Borough & Spatial Domain":
    st.subheader("Borough & Spatial Safety Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        borough_summary = filtered_df.groupby("borough", observed=True).agg(
            Total_Accidents=("crash_date", "count"),
            Injuries=("injured", "sum"),
            Fatalities=("killed", "sum")
        ).reset_index()
        
        borough_summary["Casualty_Rate_Pct"] = (
            (borough_summary["Injuries"] + borough_summary["Fatalities"]) / 
            borough_summary["Total_Accidents"] * 100
        ).round(2)
        
        fig_b_tot = px.bar(
            borough_summary.sort_values("Total_Accidents", ascending=False),
            x="borough",
            y="Total_Accidents",
            color="borough",
            color_discrete_map=BOROUGH_COLORS,
            text_auto=True,
            title="Total Collision Volume by Borough"
        )
        fig_b_tot.update_layout(showlegend=False)
        st.plotly_chart(format_fig(fig_b_tot, "Total Collisions per Borough", height=380), use_container_width=True)
        
    with col2:
        fig_cas = px.bar(
            borough_summary.sort_values("Casualty_Rate_Pct", ascending=False),
            x="borough",
            y="Casualty_Rate_Pct",
            color="borough",
            color_discrete_map=BOROUGH_COLORS,
            text_auto=".1f",
            title="Casualty Rate (% of crashes resulting in Injury or Fatality)"
        )
        fig_cas.update_layout(showlegend=False, yaxis_title="Casualty Rate (%)")
        st.plotly_chart(format_fig(fig_cas, "Borough Casualty Rates (% of Crashes)", height=380), use_container_width=True)

    # Stacked Severity within Boroughs
    st.markdown("#### Severity Composition Across Boroughs")
    b_sev = filtered_df.groupby(["borough", "severity"], observed=True).size().reset_index(name="Accidents")
    
    fig_b_stack = px.bar(
        b_sev,
        x="borough",
        y="Accidents",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        barmode="stack",
        text_auto=True
    )
    st.plotly_chart(format_fig(fig_b_stack, "Accident Severity Composition per Borough", height=400), use_container_width=True)

    # Summary table
    st.markdown("#### Borough Safety Performance Metric Matrix")
    st.dataframe(
        borough_summary.sort_values("Total_Accidents", ascending=False).rename(
            columns={
                "borough": "Borough",
                "Total_Accidents": "Total Collisions",
                "Injuries": "Injuries",
                "Fatalities": "Fatalities",
                "Casualty_Rate_Pct": "Casualty Rate (%)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# DOMAIN 3: TEMPORAL & TIME DYNAMICS
# =========================================================
elif selected_domain == "Temporal & Time Dynamics":
    st.subheader("Temporal Trends & Time-Based Collision Dynamics")
    
    t1, t2 = st.columns(2)
    
    with t1:
        yearly = filtered_df.groupby("year").size().reset_index(name="Accidents")
        yearly["year_str"] = yearly["year"].astype(str)
        
        fig_yr = px.line(
            yearly,
            x="year_str",
            y="Accidents",
            markers=True,
            line_shape="spline"
        )
        fig_yr.update_traces(line_color=ACCENT_BLUE, line_width=3, marker=dict(size=8))
        fig_yr.update_xaxes(title="Year")
        fig_yr.update_yaxes(title="Total Collisions")
        st.plotly_chart(format_fig(fig_yr, "Yearly Collision Trend Trajectory", height=360), use_container_width=True)

    with t2:
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = filtered_df["month_name"].value_counts().reindex(month_order).fillna(0).reset_index()
        monthly.columns = ["Month", "Accidents"]
        
        fig_mo = px.bar(
            monthly,
            x="Month",
            y="Accidents",
            color_discrete_sequence=[ACCENT_PURPLE],
            text_auto=True
        )
        st.plotly_chart(format_fig(fig_mo, "Monthly Seasonality Distribution", height=360), use_container_width=True)

    # Day of Week & Hourly Profile
    d1, d2 = st.columns(2)
    
    with d1:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = filtered_df["day_of_week"].value_counts().reindex(day_order).fillna(0).reset_index()
        dow_counts.columns = ["Day", "Accidents"]
        
        fig_dow = px.bar(
            dow_counts,
            x="Day",
            y="Accidents",
            color_discrete_sequence=["#38bdf8"],
            text_auto=True
        )
        st.plotly_chart(format_fig(fig_dow, "Collision Frequency by Day of Week", height=360), use_container_width=True)
        
    with d2:
        hourly_counts = filtered_df["crash_hour"].value_counts().reindex(range(24)).fillna(0).reset_index()
        hourly_counts.columns = ["Hour", "Accidents"]
        
        fig_hr = px.bar(
            hourly_counts,
            x="Hour",
            y="Accidents",
            color_discrete_sequence=["#f59e0b"],
            text_auto=True
        )
        fig_hr.update_xaxes(tickmode="linear", dtick=2, title="Hour of the Day (24h)")
        st.plotly_chart(format_fig(fig_hr, "Crash Distribution across 24 Hours", height=360), use_container_width=True)

    # Interactive Day x Hour Collision Risk Heatmap
    st.markdown("#### Crash Risk Matrix: Day of Week vs. Hour of Day")
    heatmap_data = pd.crosstab(
        filtered_df["day_of_week"],
        filtered_df["crash_hour"]
    ).reindex(index=day_order, columns=range(24), fill_value=0)
    
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Collision Count"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    st.plotly_chart(format_fig(fig_heat, "Collision Density Heatmap (Day × Hour)", height=380), use_container_width=True)

# =========================================================
# DOMAIN 4: SEVERITY & CASUALTIES
# =========================================================
elif selected_domain == "Severity & Casualties":
    st.subheader("Accident Severity & Casualty Deep-Dive")
    
    s1, s2, s3 = st.columns(3)
    s1.metric("Property Damage Crashes", f"{property_damage_count:,}", f"{property_damage_count/total_accidents*100:.1f}% of total")
    s2.metric("Injury Collisions", f"{injury_record_count:,}", f"{injury_record_count/total_accidents*100:.1f}% of total", delta_color="inverse")
    s3.metric("Fatal Crashes", f"{fatal_record_count:,}", f"{fatal_record_count/total_accidents*100:.2f}% of total", delta_color="inverse")
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        sev_df = pd.DataFrame({
            "Severity": ["Property Damage", "Injury", "Fatal"],
            "Count": [property_damage_count, injury_record_count, fatal_record_count],
            "Injuries_Sum": [0, total_injuries, 0],
            "Fatalities_Sum": [0, 0, total_fatalities]
        })
        fig_sev_bar = px.bar(
            sev_df,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            text_auto=True,
            title="Total Collision Counts by Severity"
        )
        st.plotly_chart(format_fig(fig_sev_bar, "Accident Counts by Severity Category", height=380), use_container_width=True)
        
    with col_b:
        cas_df = pd.DataFrame({
            "Category": ["Persons Injured", "Persons Killed"],
            "Casualties": [total_injuries, total_fatalities]
        })
        fig_cas_pie = px.pie(
            cas_df,
            names="Category",
            values="Casualties",
            hole=0.5,
            color="Category",
            color_discrete_map={"Persons Injured": COLOR_INJURY, "Persons Killed": COLOR_FATAL}
        )
        fig_cas_pie.update_traces(textposition="inside", textinfo="percent+value")
        st.plotly_chart(format_fig(fig_cas_pie, "Human Casualty Breakdown", height=380), use_container_width=True)

    # Severity by Hour
    st.markdown("#### Hourly Severity Profile")
    hourly_sev = filtered_df.groupby(["crash_hour", "severity"], observed=True).size().reset_index(name="Accidents")
    
    fig_hr_sev = px.line(
        hourly_sev,
        x="crash_hour",
        y="Accidents",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        markers=True
    )
    fig_hr_sev.update_xaxes(tickmode="linear", dtick=2, title="Hour of Day (0 - 23)")
    st.plotly_chart(format_fig(fig_hr_sev, "Hourly Collision Trends by Severity Level", height=380), use_container_width=True)

# =========================================================
# DOMAIN 5: VEHICLES & CONTRIBUTING FACTORS
# =========================================================
elif selected_domain == "Vehicles & Contributing Factors":
    st.subheader("Vehicle Categories & Contributing Risk Factors")
    
    v1, v2 = st.columns(2)
    
    with v1:
        st.markdown("#### Top 10 Reported Contributing Factors")
        factors = filtered_df["factor"].loc[
            ~filtered_df["factor"].astype(str).str.lower().isin(["unspecified", "unknown", ""])
        ].value_counts().head(10).reset_index()
        factors.columns = ["Factor", "Accidents"]
        
        if not factors.empty:
            fig_fact = px.bar(
                factors,
                x="Accidents",
                y="Factor",
                orientation="h",
                color_discrete_sequence=[COLOR_INJURY],
                text_auto=True
            )
            fig_fact.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(format_fig(fig_fact, "Top 10 Collision Contributing Causes", height=420), use_container_width=True)
        else:
            st.info("No specific contributing factor records found for this slice.")
            
    with v2:
        st.markdown("#### Top 10 Vehicle Types Involved")
        vehicles = filtered_df["vehicle"].loc[
            ~filtered_df["vehicle"].astype(str).str.lower().isin(["unknown", "unspecified", ""])
        ].value_counts().head(10).reset_index()
        vehicles.columns = ["Vehicle_Type", "Accidents"]
        
        if not vehicles.empty:
            fig_veh = px.bar(
                vehicles,
                x="Accidents",
                y="Vehicle_Type",
                orientation="h",
                color_discrete_sequence=[ACCENT_BLUE],
                text_auto=True
            )
            fig_veh.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(format_fig(fig_veh, "Top 10 Vehicle Types in Collisions", height=420), use_container_width=True)
        else:
            st.info("No specific vehicle type records found for this slice.")

    # High-Risk Factors in Fatal & Injury Accidents
    st.markdown("#### Fatal & Injury Collision Primary Causes")
    severe_factors = filtered_df.loc[
        filtered_df["severity"].isin(["Injury", "Fatal"]) &
        ~filtered_df["factor"].astype(str).str.lower().isin(["unspecified", "unknown", ""])
    ]["factor"].value_counts().head(8).reset_index()
    severe_factors.columns = ["Contributing Factor", "Casualty Crashes"]
    
    if not severe_factors.empty:
        fig_sev_fac = px.bar(
            severe_factors,
            x="Casualty Crashes",
            y="Contributing Factor",
            orientation="h",
            color_discrete_sequence=[COLOR_FATAL],
            text_auto=True
        )
        fig_sev_fac.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(format_fig(fig_sev_fac, "Top Factors Producing Casualties (Injuries + Fatalities)", height=360), use_container_width=True)

# =========================================================
# DOMAIN 6: GEOSPATIAL MAPS & HOTSPOTS
# =========================================================
elif selected_domain == "Geospatial Maps & Hotspots":
    st.subheader("Geospatial Accident Distribution & DBSCAN Hotspot Detection")
    
    map_tabs = st.tabs([
        "Interactive NYC Scatter Map",
        "DBSCAN Hotspot Cluster Map",
        "NYC Accident Density Map",
        "Hotspot Cluster Rankings Table"
    ])
    
    # Valid coordinates filter
    geo_df = filtered_df.dropna(subset=["latitude", "longitude"])
    geo_df = geo_df[
        geo_df["latitude"].between(40.45, 41.05) & 
        geo_df["longitude"].between(-74.35, -73.65)
    ]
    
    with map_tabs[0]:
        st.markdown("#### Interactive Collision Map (NYC Coordinates)")
        
        map_c1, map_c2 = st.columns([3, 1])
        with map_c2:
            map_pt_limit = st.slider("Map Sampling Points", min_value=1000, max_value=25000, value=8000, step=1000)
            map_color_choice = st.selectbox("Color Pins By:", ["severity", "borough"], format_func=lambda x: "Accident Severity" if x == "severity" else "Borough")
        
        with map_c1:
            st.caption(f"Showing up to {map_pt_limit:,} points out of {len(geo_df):,} valid coordinates.")
        
        if not geo_df.empty:
            sample_size = min(map_pt_limit, len(geo_df))
            map_sample = geo_df.sample(n=sample_size, random_state=42).copy()
            map_sample["date_str"] = map_sample["crash_date"].dt.strftime("%Y-%m-%d")
            map_sample["hour_str"] = map_sample["crash_hour"].apply(lambda h: f"{int(h):02d}:00")
            
            color_map = SEVERITY_COLORS if map_color_choice == "severity" else BOROUGH_COLORS
            
            center_lat = float(map_sample["latitude"].mean()) if not map_sample.empty else 40.7282
            center_lon = float(map_sample["longitude"].mean()) if not map_sample.empty else -73.9450
            zoom_lvl = 11.2 if len(selected_boroughs) == 1 and selected_boroughs[0] != "Unknown" else 9.8
            
            fig_map = px.scatter_map(
                map_sample,
                lat="latitude",
                lon="longitude",
                color=map_color_choice,
                color_discrete_map=color_map,
                hover_name="borough",
                hover_data={
                    "latitude": False,
                    "longitude": False,
                    "severity": True,
                    "date_str": True,
                    "hour_str": True,
                    "factor": True
                },
                zoom=zoom_lvl,
                center=dict(lat=center_lat, lon=center_lon),
                map_style="open-street-map",
                height=620
            )
            fig_map.update_traces(marker=dict(size=6, opacity=0.75))
            fig_map.update_layout(
                paper_bgcolor=SURFACE_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(0,0,0,0.5)" if DARK_MODE else "rgba(255,255,255,0.7)"
                )
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
        else:
            st.warning("No valid NYC geographical coordinates found for current filter settings.")

    with map_tabs[1]:
        st.markdown("#### DBSCAN Density-Based Hotspot Clustering Map")
        st.caption("Pre-computed spatial accident density clusters generated using machine learning DBSCAN clustering algorithm.")
        
        hotspot_path = PROJECT_ROOT / "visualizations" / "maps" / "accident_hotspot_map.html"
        if hotspot_path.is_file():
            try:
                html_code = load_html_content(str(hotspot_path))
                components.html(html_code, height=650, scrolling=True)
            except Exception as e:
                st.error(f"Error rendering Hotspot Map: {e}")
        else:
            st.info(f"Hotspot map file not found at: {hotspot_path}")

    with map_tabs[2]:
        st.markdown("#### NYC Overall Accident Density Distribution")
        dist_path = PROJECT_ROOT / "visualizations" / "maps" / "accident_distribution_map.html"
        if dist_path.is_file():
            try:
                html_code = load_html_content(str(dist_path))
                components.html(html_code, height=650, scrolling=True)
            except Exception as e:
                st.error(f"Error rendering Distribution Map: {e}")
        else:
            st.info(f"Accident distribution map file not found at: {dist_path}")

    with map_tabs[3]:
        st.markdown("#### High-Risk Accident Hotspot Rankings")
        ranking_file = PROJECT_ROOT / "visualizations" / "reports" / "accident_hotspot_ranking.csv"
        if not ranking_file.exists():
            ranking_file = PROJECT_ROOT / "visualizations" / "reports" / "hotspot_ranking.csv"
            
        if ranking_file.exists():
            rank_df = pd.read_csv(ranking_file)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        else:
            st.info("Hotspot ranking report CSV not found.")

# =========================================================
# DOMAIN 7: ML SEVERITY PREDICTOR & EVALUATION
# =========================================================
elif selected_domain == "ML Severity Predictor & Evaluation":
    st.subheader("Machine Learning Accident Severity Prediction & Evaluation")
    
    ml_tabs = st.tabs([
        "Real-Time Severity Predictor",
        "Model Performance Metrics",
        "Feature Importance Analysis"
    ])
    
    with ml_tabs[0]:
        st.markdown("#### Predict Collision Severity Using Trained Random Forest Pipeline")
        st.caption("Input crash characteristics to calculate estimated probabilities for Property Damage, Injury, or Fatality.")
        
        @st.cache_resource
        def load_rf_model():
            import joblib
            import urllib.request
            model_path = PROJECT_ROOT / "models" / "accident_severity_random_forest.pkl"
            
            # If not present, automatically download from fallback repository
            if not model_path.exists() or model_path.stat().st_size < 1000:
                fallback_urls = [
                    "https://github.com/pavannaik081-sketch/smart-traffic-accident-intelligence/raw/main/models/accident_severity_random_forest.pkl",
                    "https://huggingface.co/datasets/pavan-data-lab/nyc-traffic-accidents/resolve/main/accident_severity_random_forest.pkl"
                ]
                model_path.parent.mkdir(parents=True, exist_ok=True)
                for u in fallback_urls:
                    try:
                        urllib.request.urlretrieve(u, str(model_path))
                        if model_path.exists() and model_path.stat().st_size > 1000:
                            break
                    except Exception:
                        pass
                        
            if model_path.exists() and model_path.stat().st_size > 1000:
                try:
                    return joblib.load(model_path)
                except Exception as ex:
                    st.error(f"Error initializing prediction model: {ex}")
                    return None
            return None

        model_pipeline = load_rf_model()
        
        if model_pipeline is not None:
            with st.form("severity_prediction_form"):
                pcol1, pcol2, pcol3 = st.columns(3)
                
                with pcol1:
                    input_borough = st.selectbox("Borough", options=["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"])
                    input_hour = st.slider("Crash Hour (0-23)", min_value=0, max_value=23, value=17)
                    input_month = st.selectbox("Month", options=list(range(1, 13)), format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1], index=9)
                    
                with pcol2:
                    input_day = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda d: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d], index=4)
                    input_year = st.selectbox("Year", options=[2023, 2024, 2025, 2026], index=1)
                    input_zip = st.text_input("Zip Code (optional)", value="10001")
                    
                with pcol3:
                    top_factors_list = [
                        "Driver Inattention/Distraction",
                        "Failure to Yield Right-of-Way",
                        "Following Too Closely",
                        "Passing or Lane Usage Improper",
                        "Unsafe Speed",
                        "Traffic Control Disregarded",
                        "Alcohol Involvement",
                        "Unspecified"
                    ]
                    input_factor = st.selectbox("Contributing Factor", options=top_factors_list)
                    
                    top_vehicles_list = [
                        "Sedan",
                        "Station Wagon/Sport Utility Vehicle",
                        "Passenger Vehicle",
                        "Taxi",
                        "Pick-up Truck",
                        "Bus",
                        "Motorcycle",
                        "Unknown"
                    ]
                    input_vehicle = st.selectbox("Vehicle Type", options=top_vehicles_list)
                    
                submit_pred = st.form_submit_button("Run Severity Prediction", use_container_width=True)
                
            if submit_pred:
                hour_sin = np.sin(2 * np.pi * input_hour / 24.0)
                hour_cos = np.cos(2 * np.pi * input_hour / 24.0)
                month_sin = np.sin(2 * np.pi * input_month / 12.0)
                month_cos = np.cos(2 * np.pi * input_month / 12.0)
                
                sample_input = pd.DataFrame([{
                    "crash_hour": input_hour,
                    "crash_year": input_year,
                    "crash_month": input_month,
                    "day_of_week": input_day,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "month_sin": month_sin,
                    "month_cos": month_cos,
                    "borough": input_borough,
                    "zip_code": input_zip if input_zip else "10001",
                    "factor": input_factor,
                    "vehicle": input_vehicle
                }])
                
                try:
                    pred_label = model_pipeline.predict(sample_input)[0]
                    pred_probas = model_pipeline.predict_proba(sample_input)[0]
                    class_labels = model_pipeline.classes_
                    proba_dict = dict(zip(class_labels, pred_probas))
                    
                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        badge_color = COLOR_FATAL if pred_label == "Fatal" else (COLOR_INJURY if pred_label == "Injury" else COLOR_DAMAGE)
                        st.markdown(f"""
                        <div style="background-color: {SURFACE_CARD}; border: 2px solid {badge_color}; border-radius: 12px; padding: 22px; text-align: center;">
                            <div style="color: {TEXT_MUTED}; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;">PREDICTED SEVERITY</div>
                            <div style="color: {badge_color}; margin: 8px 0; font-size: 34px; font-weight: 800;">{pred_label}</div>
                            <div style="color: {TEXT_MUTED}; font-size: 13px;">Calculated by Random Forest Ensemble</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        prob_df = pd.DataFrame({
                            "Severity": class_labels,
                            "Probability": [proba_dict.get(c, 0.0) * 100 for c in class_labels]
                        })
                        fig_prob = px.bar(
                            prob_df,
                            x="Probability",
                            y="Severity",
                            orientation="h",
                            color="Severity",
                            color_discrete_map=SEVERITY_COLORS,
                            text_auto=".1f",
                            title="Class Probability Distribution (%)"
                        )
                        fig_prob.update_layout(xaxis=dict(range=[0, 100], title="Probability (%)"))
                        st.plotly_chart(format_fig(fig_prob, "Prediction Confidence Probabilities", height=220), use_container_width=True)
                        
                except Exception as ex:
                    st.error(f"Prediction computation failed: {ex}")
        else:
            st.warning("Machine learning model file (models/accident_severity_random_forest.pkl) not found.")

    with ml_tabs[1]:
        st.markdown("#### Model Performance & Validation Benchmarks")
        eval_csv = PROJECT_ROOT / "visualizations" / "reports" / "phase6_model_evaluation.csv"
        comp_csv = PROJECT_ROOT / "visualizations" / "reports" / "model_comparison.csv"
        
        if eval_csv.exists():
            st.dataframe(pd.read_csv(eval_csv), use_container_width=True, hide_index=True)
        elif comp_csv.exists():
            st.dataframe(pd.read_csv(comp_csv), use_container_width=True, hide_index=True)
            
        m_col1, m_col2 = st.columns(2)
        comp_img = PROJECT_ROOT / "visualizations" / "charts" / "phase6_model_comparison.png"
        cm_path = PROJECT_ROOT / "visualizations" / "charts" / "random_forest_confusion_matrix.png"
        
        with m_col1:
            if comp_img.exists():
                st.image(str(comp_img), caption="Model Performance Comparison Benchmark", use_container_width=True)
        with m_col2:
            if cm_path.exists():
                st.image(str(cm_path), caption="Random Forest Test Set Confusion Matrix", use_container_width=True)

    with ml_tabs[2]:
        st.markdown("#### Random Forest Feature Importance")
        fi_csv = PROJECT_ROOT / "visualizations" / "reports" / "feature_importance.csv"
        if fi_csv.exists():
            fi_df = pd.read_csv(fi_csv).sort_values("Importance", ascending=False).head(15)
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=[ACCENT_BLUE],
                text_auto=".3f"
            )
            fig_fi.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(format_fig(fig_fi, "Top 15 Most Influential Predictive Features", height=450), use_container_width=True)
        else:
            st.info("Feature importance report CSV not found.")

# =========================================================
# DOMAIN 8: ACCIDENT DATA EXPLORER
# =========================================================
elif selected_domain == "Accident Data Explorer":
    st.subheader("Interactive Accident Records Explorer & Export")
    
    disp_cols = [
        "crash_date", "crash_time", "borough", "severity",
        "injured", "killed", "factor", "vehicle", "latitude", "longitude"
    ]
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        search_query = st.text_input("Search records (borough, factor, vehicle, severity...)", placeholder="Type to filter...")
    with col_s2:
        max_rows = st.select_slider("Display Row Limit", options=[100, 500, 1000, 2500, 5000], value=1000)
        
    chosen_cols = st.multiselect("Columns to view:", options=disp_cols, default=disp_cols)
    
    explorer_data = filtered_df[chosen_cols].copy()
    
    if search_query.strip():
        q = search_query.strip().lower()
        search_mask = explorer_data.astype(str).apply(
            lambda col: col.str.lower().str.contains(q, na=False, regex=False)
        ).any(axis=1)
        explorer_data = explorer_data.loc[search_mask]
        
    st.caption(f"Displaying top {min(len(explorer_data), max_rows):,} rows out of {len(explorer_data):,} matching records.")
    
    st.dataframe(
        explorer_data.head(max_rows),
        use_container_width=True,
        hide_index=True
    )
    
    # CSV Download Button
    st.markdown("#### Export Data")
    csv_bytes = explorer_data.head(50000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Filtered Records as CSV",
        data=csv_bytes,
        file_name="nyc_traffic_accidents_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# 8. FOOTER
# =========================================================
st.markdown(f"<hr style='margin-top: 35px; margin-bottom: 20px; border-color: {BORDER_COLOR};'>", unsafe_allow_html=True)

st.markdown(f"""
<div style="
    text-align: center;
    color: {TEXT_MUTED};
    background-color: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 18px;
    font-size: 13px;
">
    <strong style="color: {TEXT_MAIN}; font-size: 14px;">NYC Smart Traffic & Accident Intelligence System</strong><br>
    Powered by Python, Streamlit, Pandas, Plotly Express & Scikit-Learn Random Forest Pipeline<br>
    <span style="font-size: 12px; opacity: 0.85;">Data Source: NYC OpenData Motor Vehicle Collisions (Crashes)</span>
</div>
""", unsafe_allow_html=True)
