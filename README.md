# 🚦 NYC Smart Traffic & Accident Intelligence System

An advanced data analytics, geospatial visualization, and machine learning decision-support platform built on the NYC Motor Vehicle Collisions dataset.

---

## 🌟 Key Features & Domain Architecture

The dashboard is structured into **8 specialized analytical domains**, easily accessible via top navigation pills or the sidebar navigator:

1. **📊 Overview & KPIs**: Executive summary metrics (total collisions, casualty count, fatality rate, severity proportions, and 24-hour activity curve).
2. **🏙️ Borough & Spatial Domain**: Spatial comparison of collision volume, injury/fatality casualty rates, and severity distribution across all NYC boroughs (Brooklyn, Queens, Manhattan, Bronx, Staten Island).
3. **⏱️ Temporal & Time Dynamics**: Multi-scale temporal analysis covering year-over-year trends, monthly seasonality, 24-hour crash curves, day-of-week patterns, and an **interactive Day × Hour crash risk density heatmap**.
4. **💥 Severity & Casualties**: Granular deep-dive into collision severity levels (Property Damage, Injury, Fatal) and human casualties.
5. **🚗 Vehicles & Contributing Factors**: Top 10 reported collision causes, primary vehicle categories involved, and high-risk casualty factor analysis.
6. **🗺️ Geospatial Maps & Hotspots**:
   - **Interactive NYC Scatter Map**: Plotly OpenStreetMap pinpoints dynamically colored by severity or borough with auto-centering and zoom.
   - **DBSCAN Hotspot Cluster Map**: Folium-rendered machine learning spatial density clusters.
   - **Accident Density Distribution Map**: City-wide collision density surface.
   - **Hotspot Rankings Table**: Ranked clusters with coordinates, spread, and collision intensity.
7. **🤖 ML Severity Predictor & Evaluation**:
   - **Live Interactive Prediction Tool**: Enter crash characteristics (borough, hour, vehicle, cause) and run real-time inference via the trained Random Forest ensemble pipeline.
   - **Model Evaluation Benchmarks**: Accuracy, Balanced Accuracy, Precision, Recall, and F1-Scores.
   - **Confusion Matrix & Feature Importance**: Visual analysis of the predictive weights.
8. **📋 Accident Data Explorer**: Searchable, filterable interactive data table with custom column selection, pagination limits, and 1-click CSV export.

---

## 🚀 How to Run the Dashboard

Make sure your virtual environment is active, then launch the app using any of the following commands:

```bash
# Recommended command:
streamlit run app.py

# Or run directly from dashboard directory:
streamlit run dashboard/app.py

# Or via streamlit_app.py:
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit & Domain Tests

A comprehensive unit test suite is included in `tests/test_dashboard_domains.py` to ensure all domains, data slicing, ML pipelines, and geospatial assets function without errors:

```bash
python -m unittest tests/test_dashboard_domains.py
```

---

## 📂 Project Structure

```text
DSV-project/
├── app.py                      # Main entrypoint wrapper
├── streamlit_app.py            # Streamlit Cloud deployment entrypoint
├── requirements.txt            # Python dependencies
├── dashboard/
│   └── app.py                  # Full modern 8-domain Streamlit dashboard
├── dataset/
│   ├── processed/              # Cleaned & optimized CSV files
│   └── raw/                    # Original NYC collision records
├── models/
│   └── accident_severity_random_forest.pkl # Trained Scikit-Learn pipeline
├── visualizations/
│   ├── maps/                   # Folium DBSCAN & density map HTML files
│   ├── charts/                 # Evaluation & visual analysis charts
│   └── reports/                # Hotspot rankings & model evaluation metrics
└── tests/
    └── test_dashboard_domains.py # Automated domain validation test suite
```
