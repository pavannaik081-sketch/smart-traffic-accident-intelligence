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
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] input {
    color: #172554 !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #64748b !important;
    opacity: 1;
}
[data-testid="stMetricLabel"] {
    color: #475569 !important;
}
[data-testid="stMetricValue"] {
    color: #172554 !important;
}
.stApp [data-testid="stTab"] {
    color: #334155 !important;
}
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stDataFrame"] {
    color: #1e293b !important;
}
button[kind="primary"] {
    color: #ffffff !important;
}
button[kind="secondary"] {
    color: #172554 !important;
}
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle {
    fill: #172554 !important;
}
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button {
    background-color: #172554 !important;
    color: #ffffff !important;
    border: 1px solid #172554 !important;
}
[data-testid="stDownloadButton"] button *,
[data-testid="stButton"] button * {
    color: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button *,
[data-testid="stHeader"] button,
[data-testid="stHeader"] button * {
    color: #ffffff !important;
    opacity: 1 !important;
}
.stApp .dashboard-footer,
.stApp .dashboard-footer p,
.stApp .dashboard-footer h4 {
    color: #334155 !important;
    opacity: 1 !important;
}
.js-plotly-plot .legend text,
.js-plotly-plot .gtitle,
.js-plotly-plot .xtitle,
.js-plotly-plot .ytitle,
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text {
    fill: #172554 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SHARED PLOTLY STYLE
# =========================================================
def style_chart(fig):
    """Apply readable text and consistent chart styling."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#172554"
        ),
        title=dict(
            font=dict(size=18, color="#172554"),
            x=0.02,
            xanchor="left"
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            font=dict(color="#172554", size=12),
            title_font=dict(color="#172554")
        ),
        xaxis=dict(
            title_font=dict(color="#172554", size=13),
            tickfont=dict(color="#334155", size=11),
            gridcolor="#e5eaf2",
            zerolinecolor="#cbd5e1"
        ),
        yaxis=dict(
            title_font=dict(color="#172554", size=13),
            tickfont=dict(color="#334155", size=11),
            gridcolor="#e5eaf2",
            zerolinecolor="#cbd5e1"
        ),
        margin=dict(l=35, r=25, t=65, b=45),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#94a3b8",
            font=dict(color="#172554", size=12)
        )
    )
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(color="#172554")
    )
    return fig

# =========================================================
# 3. PROJECT PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "traffic_accidents.csv"
)

HOTSPOT_MAP_PATH = (
    PROJECT_ROOT
    / "visualizations"
    / "maps"
    / "accident_hotspot_map.html"
)

# =========================================================
# 4. LOAD DATASET
# =========================================================
@st.cache_data(show_spinner="Loading accident dataset...")
def load_data(file_path):
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
        return (
            column.strip()
            .lower()
            .replace(" ", "_")
        )

    df = pd.read_csv(
        file_path,
        low_memory=False,
        usecols=lambda column: (
            normalize_column(column) in required_columns
        )
    )

    df.columns = [
        normalize_column(column)
        for column in df.columns
    ]

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
        .astype(str)
    )

    df["factor"] = (
        df["contributing_factor_vehicle_1"]
        .fillna("Unspecified")
        .astype(str)
    )

    df["vehicle"] = (
        df["vehicle_type_code_1"]
        .fillna("Unknown")
        .astype(str)
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
# 5. DASHBOARD HEADER
# =========================================================
st.title("🚦 Smart Traffic & Accident Intelligence System")

st.markdown(
    """
    <p class="dashboard-subtitle">
    Accident Analysis | Severity Insights |
    Geographic Accident Hotspots
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 6. CHECK AND LOAD DATASET
# =========================================================
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Make sure traffic_accidents.csv exists inside "
        "dataset/raw/."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

st.sidebar.success(
    f"Dataset loaded: {len(df):,} records"
)

# =========================================================
# 7. SIDEBAR FILTERS
# =========================================================
st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.write(
    "Choose filters to explore accident records."
)

valid_dates = df["crash_date"].dropna()

if valid_dates.empty:
    st.error("No valid accident dates found in the dataset.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

date_range = st.sidebar.date_input(
    "Accident Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# FIX: Safely handle date_input returning a tuple,
# a single date, or a partially selected date range.
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

start_date = pd.Timestamp(start_date).date()
end_date = pd.Timestamp(end_date).date()

if start_date > end_date:
    start_date, end_date = end_date, start_date

borough_options = sorted(
    df["borough"].dropna().unique().tolist()
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
# 8. FILTER DATA
# =========================================================
# FIX: Compare date objects with date objects.
crash_dates = df["crash_date"].dt.date

date_mask = (
    crash_dates.ge(start_date)
    & crash_dates.le(end_date)
)

filtered_df = df[
    date_mask
    & df["borough"].isin(selected_boroughs)
    & df["severity"].isin(selected_severities)
].copy()

if filtered_df.empty:
    st.warning(
        "No accident records found for the selected filters."
    )
    st.stop()

# =========================================================
# 9. INTERACTIVE DISPLAY SETTINGS
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
        "Accident Severity" if value == "severity"
        else "Borough"
    )
)

# =========================================================
# 10. OVERALL ACCIDENT STATISTICS
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

with col1:
    st.metric(
        "Total Accidents",
        f"{total_accidents:,}"
    )

with col2:
    st.metric(
        "Total Injuries",
        f"{total_injuries:,}"
    )

with col3:
    st.metric(
        "Total Fatalities",
        f"{total_fatalities:,}"
    )

with col4:
    st.metric(
        "Fatal Accident Records",
        f"{fatal_records:,}"
    )

st.caption(
    "Fatalities count people killed; fatal accident records count "
    "accident rows classified as Fatal."
)

# =========================================================
# 11. ACCIDENT SEVERITY ANALYSIS
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# 12. TIME-BASED ACCIDENT ANALYSIS
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
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
            month=int(value),
            year=2024,
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# 13. HOURLY ACCIDENT ANALYSIS
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

fig.update_yaxes(
    title="Number of Accidents"
)

fig = style_chart(fig)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 14. DAY-WISE ACCIDENT ANALYSIS
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

fig = style_chart(fig)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 15. BOROUGH-WISE ACCIDENT ANALYSIS
# =========================================================
st.divider()
st.header("5. Borough-wise Accident Analysis")

borough_data = (
    filtered_df.groupby("borough")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="Accidents")
)

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

fig = style_chart(fig)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# 16. TOP CONTRIBUTING FACTORS
# =========================================================
st.divider()
st.header("6. Top Contributing Factors")

factor_data = (
    filtered_df[
        filtered_df["factor"].str.lower() != "unspecified"
    ]["factor"]
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True
        }
    )
else:
    st.info("No contributing-factor records available.")

# =========================================================
# 17. VEHICLE TYPE ANALYSIS
# =========================================================
st.divider()
st.header("7. Vehicle Type Analysis")

vehicle_data = (
    filtered_df[
        filtered_df["vehicle"].str.lower() != "unknown"
    ]["vehicle"]
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

    fig = style_chart(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True
        }
    )
else:
    st.info("No vehicle-type records available.")

# =========================================================
# 18. GEOGRAPHICAL ACCIDENT MAP
# =========================================================
st.divider()
st.header("8. Geographical Accident Map")

st.caption(
    "Recorded accident locations across New York City. "
    "This map does not show live traffic conditions."
)

map_df = filtered_df.dropna(
    subset=["latitude", "longitude"]
).copy()

map_df = map_df[
    map_df["latitude"].between(40, 42)
    & map_df["longitude"].between(-75, -73)
].copy()

st.write(
    f"Valid geographical accident records: {len(map_df):,}"
)

if not map_df.empty:
    map_sample = map_df.sample(
        n=min(map_point_limit, len(map_df)),
        random_state=42
    ).copy()

    map_sample["crash_date_display"] = (
        map_sample["crash_date"]
        .dt.strftime("%Y-%m-%d")
        .fillna("Unknown")
        .astype(str)
    )

    map_sample["borough_display"] = (
        map_sample["borough"]
        .fillna("Unknown")
        .astype(str)
    )

    map_sample["hour_display"] = map_sample[
        "crash_hour"
    ].apply(
        lambda value: (
            f"{int(value):02d}:00"
            if pd.notna(value)
            else "Unknown"
        )
    ).astype(str)

    map_sample["severity_display"] = (
        map_sample["severity"].astype(str)
    )

    map_plot_data = pd.DataFrame({
        "latitude": map_sample["latitude"].astype("float64").to_numpy(),
        "longitude": map_sample["longitude"].astype("float64").to_numpy(),
        "severity": map_sample["severity_display"].to_numpy(dtype=str),
        "borough": map_sample["borough_display"].to_numpy(dtype=str),
        "crash_date": map_sample["crash_date_display"].to_numpy(dtype=str),
        "crash_hour": map_sample["hour_display"].to_numpy(dtype=str)
    })

    map_plot_data = map_plot_data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=["latitude", "longitude"]
    )

    if not map_plot_data.empty:
        fig = px.scatter_map(
            map_plot_data,
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

        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin={
                "r": 10,
                "t": 60,
                "l": 10,
                "b": 10
            },
            font=dict(
                color="#172554"
            ),
            legend=dict(
                font=dict(color="#172554"),
                title_font=dict(color="#172554")
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "responsive": True
            }
        )

        st.caption(
            f"Displaying {len(map_plot_data):,} accident locations "
            f"from {len(map_df):,} valid records."
        )
    else:
        st.warning(
            "No valid numerical coordinates available for the map."
        )
else:
    st.warning(
        "No valid NYC geographical coordinates were found "
        "for the selected filters."
    )

# =========================================================
# 19. DBSCAN ACCIDENT HOTSPOT MAP
# =========================================================
st.divider()
st.header("9. Accident Hotspot Detection Map")

st.caption(
    "DBSCAN-based geographical accident clusters "
    "generated during Phase 4."
)

if HOTSPOT_MAP_PATH.exists():
    try:
        with open(
            HOTSPOT_MAP_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            map_html = file.read()

        components.html(
            map_html,
            height=650,
            scrolling=True
        )
    except Exception as error:
        st.error(
            f"Unable to display hotspot map: {error}"
        )
else:
    st.info(
        "Hotspot map not found. Check "
        "visualizations/maps/accident_hotspot_map.html."
    )

# =========================================================
# 20. ACCIDENT DATA EXPLORER
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

available_display_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

search_text = st.text_input(
    "🔎 Search accident records",
    placeholder="Search borough, severity, factor, vehicle, date..."
)

selected_table_columns = st.multiselect(
    "Choose columns to display",
    options=available_display_columns,
    default=available_display_columns
)

explorer_df = filtered_df.copy()

if search_text.strip():
    search_mask = explorer_df[
        available_display_columns
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
        use_container_width=True,
        hide_index=True
    )

st.caption(
    f"Showing {min(len(explorer_df), data_table_limit):,} of "
    f"{len(explorer_df):,} matching filtered accident records."
)

# =========================================================
# 21. DOWNLOAD FILTERED DATA
# =========================================================
st.divider()
st.header("📥 Download Accident Data")

download_data = filtered_df[
    available_display_columns
].to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=download_data,
    file_name="filtered_accident_data.csv",
    mime="text/csv"
)

# =========================================================
# 22. FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="dashboard-footer" style="
    text-align:center;
    color:#334155;
    padding:20px;
">
    <h4>Smart Traffic & Accident Intelligence System</h4>
    <p>Data Science and Machine Learning Project</p>
    <p>Dataset: NYC Motor Vehicle Collisions – Crashes</p>
    <p>Built with Python, Streamlit, Pandas and Plotly</p>
</div>
""", unsafe_allow_html=True)