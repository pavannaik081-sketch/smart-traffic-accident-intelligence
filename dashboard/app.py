import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

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
# 2. THEME SELECTION
# =========================================================
st.sidebar.title("🎨 Dashboard Appearance")

theme = st.sidebar.radio(
    "Choose Dashboard Theme",
    ["Light Mode", "Dark Mode"],
    horizontal=True,
    key="dashboard_theme"
)

DARK_MODE = theme == "Dark Mode"

if DARK_MODE:
    BG = "#0e1117"
    PANEL = "#1b2230"
    SIDEBAR = "#151b26"
    TEXT = "#f1f5f9"
    MUTED = "#cbd5e1"
    BORDER = "#334155"
    INPUT_BG = "#263244"
    ACCENT = "#60a5fa"
    GRID = "#334155"
else:
    BG = "#f4f7fc"
    PANEL = "#ffffff"
    SIDEBAR = "#eaf0f8"
    TEXT = "#172554"
    MUTED = "#475569"
    BORDER = "#dbe3ef"
    INPUT_BG = "#ffffff"
    ACCENT = "#2563eb"
    GRID = "#e5eaf2"

# =========================================================
# 3. RESPONSIVE CSS AND THEME STYLING
# =========================================================
st.markdown(f"""
<style>
.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

[data-testid="stAppViewContainer"] {{
    background-color: {BG};
}}

[data-testid="stHeader"] {{
    background-color: {BG};
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 3%;
    padding-right: 3%;
    max-width: 1600px;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {TEXT} !important;
    overflow-wrap: anywhere;
}}

p, span, label, li, small, div {{
    color: inherit;
}}

.stMarkdown,
.stMarkdown p,
.stMarkdown li {{
    color: {TEXT};
}}

.dashboard-subtitle {{
    color: {MUTED} !important;
    font-size: 16px;
}}

section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR};
    border-right: 1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}

section[data-testid="stSidebar"] label {{
    font-weight: 600;
}}

[data-testid="stMetric"] {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    padding: 18px;
    border-radius: 14px;
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
}}

[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    overflow-wrap: anywhere;
}}

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stTextArea textarea {{
    background-color: {INPUT_BG} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="select"] {{
    background-color: {PANEL} !important;
    color: {TEXT} !important;
}}

[data-baseweb="menu"] *,
[data-baseweb="popover"] * {{
    color: {TEXT} !important;
}}

.stRadio label,
.stCheckbox label {{
    color: {TEXT} !important;
}}

.stButton button,
.stDownloadButton button {{
    background-color: {ACCENT} !important;
    color: white !important;
    border: none !important;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}}

.stButton button *,
.stDownloadButton button * {{
    color: white !important;
}}

.stButton button:hover,
.stDownloadButton button:hover {{
    opacity: 0.85;
}}

[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

[data-testid="stAlert"] {{
    border-radius: 10px;
}}

[data-testid="stPlotlyChart"] {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 5px;
    overflow: hidden;
}}

.stCaption,
[data-testid="stCaptionContainer"] {{
    color: {MUTED} !important;
}}

hr {{
    border-color: {BORDER} !important;
}}

@media screen and (max-width: 1024px) {{
    .block-container {{
        padding-left: 3%;
        padding-right: 3%;
    }}

    h1 {{
        font-size: 30px !important;
    }}

    h2 {{
        font-size: 25px !important;
    }}
}}

@media screen and (max-width: 768px) {{
    .block-container {{
        padding: 1rem 0.8rem 2rem 0.8rem;
    }}

    h1 {{
        font-size: 25px !important;
        line-height: 1.3 !important;
    }}

    h2 {{
        font-size: 21px !important;
        line-height: 1.35 !important;
    }}

    h3 {{
        font-size: 18px !important;
    }}

    p {{
        font-size: 14px;
        overflow-wrap: anywhere;
    }}

    [data-testid="stMetric"] {{
        padding: 12px;
        margin-bottom: 8px;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 22px !important;
    }}

    [data-testid="stPlotlyChart"] {{
        width: 100% !important;
        padding: 0;
    }}

    .stButton button,
    .stDownloadButton button {{
        width: 100%;
    }}
}}

@media screen and (max-width: 480px) {{
    .block-container {{
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }}

    h1 {{
        font-size: 22px !important;
    }}

    h2 {{
        font-size: 19px !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 19px !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 12px !important;
    }}

    .dashboard-subtitle {{
        font-size: 13px;
    }}
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. PROJECT PATHS AND DATA SOURCE
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_URL = (
    "https://huggingface.co/datasets/"
    "pavan-data-lab/nyc-traffic-accidents/resolve/main/"
    "traffic_accidents_dashboard.csv"
)

LOCAL_DATA_PATH = (
    PROJECT_ROOT / "dataset" / "raw" / "traffic_accidents.csv"
)

DATA_SOURCE = (
    str(LOCAL_DATA_PATH)
    if LOCAL_DATA_PATH.exists()
    else DATA_URL
)

HOTSPOT_MAP_PATH = (
    PROJECT_ROOT
    / "visualizations"
    / "maps"
    / "accident_hotspot_map.html"
)

# =========================================================
# 5. CHART STYLING
# =========================================================
def style_chart(fig):
    fig.update_layout(
        template="plotly_dark" if DARK_MODE else "plotly_white",
        font=dict(
            family="Arial",
            size=13,
            color=TEXT
        ),
        title=dict(
            font=dict(size=18, color=TEXT),
            x=0.02
        ),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        legend=dict(
            font=dict(color=TEXT, size=12),
            title_font=dict(color=TEXT)
        ),
        xaxis=dict(
            title_font=dict(color=TEXT),
            tickfont=dict(color=TEXT),
            gridcolor=GRID
        ),
        yaxis=dict(
            title_font=dict(color=TEXT),
            tickfont=dict(color=TEXT),
            gridcolor=GRID
        ),
        margin=dict(l=35, r=25, t=65, b=45),
        autosize=True
    )
    return fig

# =========================================================
# 6. LOAD DATASET
# =========================================================
@st.cache_data(show_spinner="Loading accident dataset...")
def load_data(file_source):

    required_columns = {
        "crash_date",
        "crash_time",
        "borough",
        "latitude",
        "longitude",
        "number_of_persons_injured",
        "number_of_persons_killed",
        "contributing_factor_vehicle_1",
        "vehicle_type_code_1"
    }

    def normalize_column(column):
        return column.strip().lower().replace(" ", "_")

    df = pd.read_csv(
        file_source,
        low_memory=False,
        usecols=lambda column: (
            normalize_column(column) in required_columns
        )
    )

    df.columns = [
        normalize_column(column)
        for column in df.columns
    ]

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["crash_date"] = pd.to_datetime(
        df["crash_date"],
        errors="coerce"
    )

    df["crash_time"] = df["crash_time"].astype("string")

    df["crash_hour"] = pd.to_numeric(
        df["crash_time"].str.extract(r"^(\d{1,2})")[0],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["injured"] = pd.to_numeric(
        df["number_of_persons_injured"],
        errors="coerce"
    ).fillna(0)

    df["killed"] = pd.to_numeric(
        df["number_of_persons_killed"],
        errors="coerce"
    ).fillna(0)

    df["borough"] = (
        df["borough"]
        .fillna("Unknown")
        .astype("category")
    )

    df["factor"] = (
        df["contributing_factor_vehicle_1"]
        .fillna("Unspecified")
        .astype("category")
    )

    df["vehicle"] = (
        df["vehicle_type_code_1"]
        .fillna("Unknown")
        .astype("category")
    )

    df["severity"] = np.select(
        [
            df["killed"] > 0,
            df["injured"] > 0
        ],
        [
            "Fatal",
            "Injury"
        ],
        default="Property Damage"
    )

    df["year"] = df["crash_date"].dt.year
    df["month"] = df["crash_date"].dt.month
    df["month_name"] = df["crash_date"].dt.strftime("%b")
    df["day_of_week"] = df["crash_date"].dt.day_name()

    return df

# =========================================================
# 7. DASHBOARD HEADER
# =========================================================
st.title("🚦 Smart Traffic & Accident Intelligence System")

st.markdown("""
<p class="dashboard-subtitle">
Accident Analysis | Severity Insights |
Geographic Accident Hotspots
</p>
""", unsafe_allow_html=True)

# =========================================================
# 8. LOAD DATA
# =========================================================
st.info(
    "Dataset source: "
    + (
        "Local CSV file"
        if DATA_SOURCE == str(LOCAL_DATA_PATH)
        else "Hugging Face"
    )
)

try:
    df = load_data(DATA_SOURCE)

except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.warning(
        "Check that your Hugging Face dataset is public "
        "and that the CSV URL is correct."
    )
    st.stop()

st.sidebar.success(
    f"Dataset loaded: {len(df):,} records"
)

# =========================================================
# 9. SIDEBAR FILTERS
# =========================================================
st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.write(
    "Choose filters to explore accident records."
)

valid_dates = df["crash_date"].dropna()

if valid_dates.empty:
    st.error("No valid accident dates found.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

date_range = st.sidebar.date_input(
    "Accident Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)):

    selected_dates = [
        value for value in date_range
        if value is not None
    ]

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date

else:
    start_date = end_date = date_range

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

if start_date > end_date:
    start_date, end_date = end_date, start_date

borough_options = sorted(
    df["borough"].dropna().astype(str).unique().tolist()
)

selected_boroughs = st.sidebar.multiselect(
    "Select Borough",
    options=borough_options,
    default=borough_options
)

severity_options = [
    "Property Damage",
    "Injury",
    "Fatal"
]

selected_severities = st.sidebar.multiselect(
    "Select Accident Severity",
    options=severity_options,
    default=severity_options
)

# =========================================================
# 10. FILTER DATA
# =========================================================
end_date_exclusive = end_date + pd.Timedelta(days=1)

date_mask = (
    df["crash_date"].ge(start_date)
    & df["crash_date"].lt(end_date_exclusive)
)

filtered_df = df.loc[
    date_mask
    & df["borough"].astype(str).isin(selected_boroughs)
    & df["severity"].isin(selected_severities)
]

if filtered_df.empty:
    st.warning(
        "No accident records found for selected filters."
    )
    st.stop()

# =========================================================
# 11. DISPLAY SETTINGS
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("⚙️ Interactive Display")

map_point_limit = st.sidebar.slider(
    "Maximum map points",
    min_value=1000,
    max_value=20000,
    value=10000,
    step=1000
)

data_table_limit = st.sidebar.slider(
    "Maximum table records",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

map_color_by = st.sidebar.selectbox(
    "Map colour based on",
    options=["severity", "borough"],
    format_func=lambda value: (
        "Accident Severity"
        if value == "severity"
        else "Borough"
    )
)

# =========================================================
# 12. OVERALL ACCIDENT STATISTICS
# =========================================================
st.divider()
st.header("📊 Overall Accident Statistics")

total_accidents = len(filtered_df)
total_injuries = int(filtered_df["injured"].sum())
total_fatalities = int(filtered_df["killed"].sum())

fatal_records = int(
    (filtered_df["severity"] == "Fatal").sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accidents", f"{total_accidents:,}")
col2.metric("Total Injuries", f"{total_injuries:,}")
col3.metric("Total Fatalities", f"{total_fatalities:,}")
col4.metric("Fatal Accident Records", f"{fatal_records:,}")

st.caption(
    "Fatalities count people killed; fatal accident records "
    "count accident rows classified as Fatal."
)

# =========================================================
# 13. ACCIDENT SEVERITY ANALYSIS
# =========================================================
st.divider()
st.header("1. Accident Severity Analysis")

severity_data = (
    filtered_df["severity"]
    .value_counts()
    .reindex(severity_options, fill_value=0)
    .reset_index()
)

severity_data.columns = ["Severity", "Accidents"]

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        severity_data,
        names="Severity",
        values="Accidents",
        hole=0.45,
        title="Accident Severity Distribution",
        color="Severity",
        color_discrete_map={
            "Property Damage": "#82c7f8",
            "Injury": "#0875d1",
            "Fatal": "#ff7777"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

with col2:
    fig = px.bar(
        severity_data,
        x="Severity",
        y="Accidents",
        color="Severity",
        title="Number of Accidents by Severity",
        text_auto=True,
        color_discrete_map={
            "Property Damage": "#82c7f8",
            "Injury": "#0875d1",
            "Fatal": "#ff7777"
        }
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

# =========================================================
# 14. TIME-BASED ACCIDENT ANALYSIS
# =========================================================
st.divider()
st.header("2. Time-Based Accident Analysis")

col1, col2 = st.columns(2)

with col1:
    yearly_data = (
        filtered_df.dropna(subset=["year"])
        .groupby("year")
        .size()
        .reset_index(name="Accidents")
    )

    yearly_data["year"] = (
        yearly_data["year"].astype(int).astype(str)
    )

    fig = px.line(
        yearly_data,
        x="year",
        y="Accidents",
        markers=True,
        title="Year-wise Accident Trend"
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

with col2:
    monthly_data = (
        filtered_df.dropna(subset=["month"])
        .groupby("month")
        .size()
        .reset_index(name="Accidents")
        .sort_values("month")
    )

    monthly_data["Month"] = monthly_data["month"].apply(
        lambda value: pd.Timestamp(
            year=2024,
            month=int(value),
            day=1
        ).strftime("%b")
    )

    fig = px.bar(
        monthly_data,
        x="Month",
        y="Accidents",
        title="Month-wise Accident Distribution",
        text_auto=True
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )

# =========================================================
# 15. HOURLY ACCIDENT ANALYSIS
# =========================================================
st.divider()
st.header("3. Hourly Accident Analysis")

hourly_data = (
    filtered_df.dropna(subset=["crash_hour"])
    .groupby("crash_hour")
    .size()
    .reset_index(name="Accidents")
)

hourly_data["Hour"] = hourly_data["crash_hour"].astype(int)

fig = px.bar(
    hourly_data,
    x="Hour",
    y="Accidents",
    title="Accidents by Hour of Day",
    text_auto=True
)

fig.update_xaxes(
    tickmode="linear",
    dtick=1,
    title="Hour of Day"
)

fig.update_yaxes(title="Number of Accidents")

st.plotly_chart(
    style_chart(fig),
    width="stretch"
)

# =========================================================
# 16. DAY-WISE ACCIDENT ANALYSIS
# =========================================================
st.divider()
st.header("4. Day-wise Accident Analysis")

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_data = (
    filtered_df["day_of_week"]
    .value_counts()
    .reindex(day_order, fill_value=0)
    .reset_index()
)

day_data.columns = ["Day", "Accidents"]

fig = px.bar(
    day_data,
    x="Day",
    y="Accidents",
    title="Accidents by Day of Week",
    text_auto=True
)

st.plotly_chart(
    style_chart(fig),
    width="stretch"
)

# =========================================================
# 17. BOROUGH-WISE ACCIDENT ANALYSIS
# =========================================================
st.divider()
st.header("5. Borough-wise Accident Analysis")

borough_data = (
    filtered_df.groupby("borough", observed=True)
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Accidents")
)

borough_data["borough"] = borough_data["borough"].astype(str)

fig = px.bar(
    borough_data,
    x="Accidents",
    y="borough",
    orientation="h",
    title="Accident Records by Borough",
    text_auto=True
)

fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="Number of Accidents",
    yaxis_title="Borough"
)

st.plotly_chart(
    style_chart(fig),
    width="stretch"
)

# =========================================================
# 18. TOP CONTRIBUTING FACTORS
# =========================================================
st.divider()
st.header("6. Top Contributing Factors")

factor_series = filtered_df["factor"].astype(str)

factor_data = (
    factor_series.loc[
        factor_series.str.lower() != "unspecified"
    ]
    .value_counts()
    .head(10)
    .rename_axis("Factor")
    .reset_index(name="Accidents")
)

if not factor_data.empty:
    fig = px.bar(
        factor_data,
        x="Accidents",
        y="Factor",
        orientation="h",
        title="Top 10 Reported Contributing Factors",
        text_auto=True
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )
else:
    st.info("No contributing-factor records available.")

# =========================================================
# 19. VEHICLE TYPE ANALYSIS
# =========================================================
st.divider()
st.header("7. Vehicle Type Analysis")

vehicle_series = filtered_df["vehicle"].astype(str)

vehicle_data = (
    vehicle_series.loc[
        vehicle_series.str.lower() != "unknown"
    ]
    .value_counts()
    .head(10)
    .rename_axis("Vehicle Type")
    .reset_index(name="Accidents")
)

if not vehicle_data.empty:
    fig = px.bar(
        vehicle_data,
        x="Accidents",
        y="Vehicle Type",
        orientation="h",
        title="Top 10 Vehicle Types Involved in Accidents"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        style_chart(fig),
        width="stretch"
    )
else:
    st.info("No vehicle-type records available.")

# =========================================================
# 20. GEOGRAPHICAL ACCIDENT MAP
# =========================================================
st.divider()
st.header("8. Geographical Accident Map")

st.caption(
    "Recorded accident locations across New York City. "
    "This map does not show live traffic conditions."
)

geo_mask = (
    filtered_df["latitude"].between(40, 42)
    & filtered_df["longitude"].between(-75, -73)
)

geo_columns = [
    "latitude",
    "longitude",
    "severity",
    "borough",
    "crash_date",
    "crash_hour"
]

map_df = filtered_df.loc[
    geo_mask,
    geo_columns
]

st.write(
    f"Valid geographical accident records: {len(map_df):,}"
)

if not map_df.empty:

    map_sample = map_df.sample(
        n=min(map_point_limit, len(map_df)),
        random_state=42
    ).copy()

    map_sample["borough"] = (
        map_sample["borough"].astype(str)
    )

    map_sample["crash_date"] = (
        map_sample["crash_date"]
        .dt.strftime("%Y-%m-%d")
        .fillna("Unknown")
    )

    map_sample["crash_hour"] = map_sample[
        "crash_hour"
    ].apply(
        lambda value: (
            f"{int(value):02d}:00"
            if pd.notna(value)
            else "Unknown"
        )
    )

    fig = px.scatter_map(
        map_sample,
        lat="latitude",
        lon="longitude",
        color=map_color_by,
        hover_name="borough",
        hover_data={
            "latitude": False,
            "longitude": False,
            "crash_date": True,
            "crash_hour": True
        },
        zoom=9,
        height=650,
        center={
            "lat": 40.7128,
            "lon": -74.0060
        },
        map_style="open-street-map",
        title="NYC Accident Locations"
    )

    fig.update_traces(
        marker={
            "size": 8,
            "opacity": 0.75
        }
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "responsive": True
        }
    )

    st.caption(
        f"Displaying {len(map_sample):,} accident locations "
        f"from {len(map_df):,} valid records."
    )

else:
    st.warning(
        "No valid NYC geographical coordinates were found."
    )

# =========================================================
# 21. DBSCAN ACCIDENT HOTSPOT MAP
# =========================================================
st.divider()
st.header("9. Accident Hotspot Detection Map")

st.caption(
    "DBSCAN-based geographical accident clusters "
    "generated during Phase 4."
)

if HOTSPOT_MAP_PATH.is_file():

    try:
        # Correct current Streamlit API:
        # Pass the local HTML Path directly.
        # Do not pass scrolling or HTML contents as a string.
        st.iframe(
            HOTSPOT_MAP_PATH,
            width="stretch",
            height=650
        )

    except Exception as error:
        st.error(
            f"Unable to display hotspot map: {error}"
        )

else:
    st.info(
        "Hotspot map not found. Check that this file exists: "
        "visualizations/maps/accident_hotspot_map.html"
    )

# =========================================================
# 22. ACCIDENT DATA EXPLORER
# =========================================================
st.divider()
st.header("10. Accident Data Explorer")

st.write(
    "Explore accident records using the selected sidebar filters."
)

display_columns = [
    "crash_date",
    "crash_time",
    "borough",
    "latitude",
    "longitude",
    "severity",
    "injured",
    "killed",
    "factor",
    "vehicle"
]

search_text = st.text_input(
    "🔎 Search accident records",
    placeholder="Search borough, severity, factor, vehicle, date..."
)

selected_table_columns = st.multiselect(
    "Choose columns to display",
    options=display_columns,
    default=display_columns
)

explorer_df = filtered_df

if search_text.strip():

    search_mask = explorer_df[
        display_columns
    ].astype(str).apply(
        lambda column: column.str.contains(
            search_text.strip(),
            case=False,
            na=False,
            regex=False
        )
    ).any(axis=1)

    explorer_df = explorer_df.loc[search_mask]

if not selected_table_columns:

    st.info(
        "Select at least one column to display accident records."
    )

else:

    st.dataframe(
        explorer_df[selected_table_columns].head(data_table_limit),
        width="stretch",
        hide_index=True
    )

st.caption(
    f"Showing {min(len(explorer_df), data_table_limit):,} of "
    f"{len(explorer_df):,} matching filtered records."
)

# =========================================================
# 23. DOWNLOAD FILTERED DATA
# =========================================================
st.divider()
st.header("📥 Download Accident Data")

DOWNLOAD_ROW_LIMIT = 100_000

st.caption(
    f"CSV export is limited to the first "
    f"{DOWNLOAD_ROW_LIMIT:,} matching records "
    "to protect cloud memory."
)

if st.button("Prepare CSV download"):

    csv_to_download = (
        filtered_df[display_columns]
        .head(DOWNLOAD_ROW_LIMIT)
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_to_download,
        file_name="filtered_accident_data.csv",
        mime="text/csv",
        width="stretch"
    )

# =========================================================
# 24. FOOTER
# =========================================================
st.divider()

st.markdown(f"""
<div style="
    text-align:center;
    color:{TEXT};
    background-color:{PANEL};
    border:1px solid {BORDER};
    border-radius:12px;
    padding:20px;
">
    <h4 style="color:{TEXT};">
        Smart Traffic & Accident Intelligence System
    </h4>
    <p style="color:{MUTED};">
        Data Science and Machine Learning Project
    </p>
    <p style="color:{MUTED};">
        Dataset: NYC Motor Vehicle Collisions – Crashes
    </p>
    <p style="color:{MUTED};">
        Built with Python, Streamlit, Pandas and Plotly
    </p>
</div>
""", unsafe_allow_html=True)
