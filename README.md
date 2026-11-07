---
title: NYC Smart Traffic & Accident Intelligence Platform
emoji: 🏙️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# NYC Smart Traffic & Accident Intelligence Platform

An advanced urban data analytics, geospatial visualization, and machine learning decision-support platform built on the NYC Motor Vehicle Collisions dataset.

---

## Analytical Domains

The dashboard features **8 specialized analytical domains**, easily accessible via top navigation pills or the sidebar navigator:

1. **Executive Overview & KPIs**: Executive summary metrics (total collisions, casualty count, fatality rate, severity proportions, and 24-hour activity curve).
2. **Borough & Spatial Domain**: Spatial comparison of collision volume, injury/fatality casualty rates, and severity distribution across all NYC boroughs (Brooklyn, Queens, Manhattan, Bronx, Staten Island).
3. **Temporal & Time Dynamics**: Multi-scale temporal analysis covering year-over-year trends, monthly seasonality, 24-hour crash curves, day-of-week patterns, and an **interactive Day × Hour crash risk density heatmap**.
4. **Severity & Casualties**: Granular deep-dive into collision severity levels (Property Damage, Injury, Fatal) and human casualties.
5. **Vehicles & Contributing Factors**: Top 10 reported collision causes, primary vehicle categories involved, and high-risk casualty factor analysis.
6. **Geospatial Maps & Hotspots**:
   - **Interactive NYC Scatter Map**: Plotly OpenStreetMap pinpoints dynamically colored by severity or borough with auto-centering and zoom.
   - **DBSCAN Hotspot Cluster Map**: Folium-rendered machine learning spatial density clusters.
   - **Accident Density Distribution Map**: City-wide collision density surface.
   - **Hotspot Rankings Table**: Ranked clusters with coordinates, spread, and collision intensity.
7. **ML Severity Predictor & Evaluation**:
   - **Live Interactive Prediction Tool**: Enter crash characteristics (borough, hour, vehicle, cause) and run real-time inference via the trained Random Forest ensemble pipeline.
   - **Model Evaluation Benchmarks**: Accuracy, Balanced Accuracy, Precision, Recall, and F1-Scores.
   - **Confusion Matrix & Feature Importance**: Visual analysis of the predictive weights.
8. **Accident Data Explorer**: Searchable, filterable interactive data table with custom column selection, pagination limits, and 1-click CSV export.

---

## Deployment Platforms

### Option 1: Hugging Face Spaces (Recommended - Free 16 GB RAM)
Hugging Face Spaces provides **16 GB RAM** (16x more than Streamlit Cloud's 1 GB free tier), eliminating any memory limits.

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Enter a Space name: `smart-traffic-accident-intelligence`.
3. Select **Streamlit** as the Space SDK.
4. Set Space hardware to **CPU Basic (Free - 16 GB RAM)**.
5. Under "Upload from your computer" or "Connect GitHub", link your GitHub repo:
   `https://github.com/pavannaik081-sketch/smart-traffic-accident-intelligence`.
6. Click **Create Space** — your app will build and be live with a permanent URL!

### Option 2: Render.com (Free Web Service)
1. Sign in to [https://render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository `pavannaik081-sketch/smart-traffic-accident-intelligence`.
4. Render automatically uses the included `render.yaml` configuration.
5. Click **Create Web Service**.

### Option 3: Streamlit Community Cloud
1. Go to your existing app page at `smart-traffic-accident-intelligence.streamlit.app`.
2. Click **"Reboot the app"** (or in Settings -> **Reboot App**).
3. The latest memory-optimized code now stays well below the 1 GB limit.

---

## Running Locally

Make sure your virtual environment is active, then launch:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Running Automated Domain Tests

A comprehensive unit test suite is included in `tests/test_dashboard_domains.py`:

```bash
python -m unittest tests/test_dashboard_domains.py
```
