import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestDashboardDomains(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Locate preprocessed or raw data
        candidates = [
            PROJECT_ROOT / "dataset" / "processed" / "traffic_accidents_dashboard.csv",
            PROJECT_ROOT / "dataset" / "processed" / "traffic_accidents_cleaned.csv",
            PROJECT_ROOT / "dataset" / "raw" / "traffic_accidents.csv"
        ]
        cls.data_path = None
        for cand in candidates:
            if cand.exists():
                cls.data_path = cand
                break
        
        if cls.data_path is not None:
            cls.df = pd.read_csv(cls.data_path, nrows=10000, low_memory=False)
            # Normalize columns
            cls.df.columns = [c.strip().lower().replace(" ", "_") for c in cls.df.columns]
            cls.df["crash_date"] = pd.to_datetime(cls.df["crash_date"], errors="coerce")
            cls.df["injured"] = pd.to_numeric(cls.df["number_of_persons_injured"], errors="coerce").fillna(0).astype("int16")
            cls.df["killed"] = pd.to_numeric(cls.df["number_of_persons_killed"], errors="coerce").fillna(0).astype("int16")
            cls.df["borough"] = cls.df["borough"].fillna("Unknown").astype(str).str.strip().str.upper()
            conditions = [cls.df["killed"] > 0, cls.df["injured"] > 0]
            cls.df["severity"] = np.select(conditions, ["Fatal", "Injury"], default="Property Damage")
        else:
            cls.df = pd.DataFrame()

    def test_dataset_exists_and_loads(self):
        self.assertIsNotNone(self.data_path, "Dataset file must exist locally")
        self.assertGreater(len(self.df), 0, "Dataset must have non-zero rows")

    def test_borough_filtering_domains(self):
        for b in ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]:
            sub = self.df[self.df["borough"] == b]
            # Verify aggregation does not throw errors
            tot = len(sub)
            inj = sub["injured"].sum()
            kil = sub["killed"].sum()
            self.assertGreaterEqual(tot, 0)
            self.assertGreaterEqual(inj, 0)
            self.assertGreaterEqual(kil, 0)

    def test_ml_severity_pipeline(self):
        model_path = PROJECT_ROOT / "models" / "accident_severity_random_forest.pkl"
        if model_path.exists():
            pipe = joblib.load(model_path)
            sample = pd.DataFrame([{
                "crash_hour": 14,
                "crash_year": 2024,
                "crash_month": 5,
                "day_of_week": 2,
                "hour_sin": np.sin(2 * np.pi * 14 / 24.0),
                "hour_cos": np.cos(2 * np.pi * 14 / 24.0),
                "month_sin": np.sin(2 * np.pi * 5 / 12.0),
                "month_cos": np.cos(2 * np.pi * 5 / 12.0),
                "borough": "BROOKLYN",
                "zip_code": "11201",
                "factor": "Driver Inattention/Distraction",
                "vehicle": "Sedan"
            }])
            pred = pipe.predict(sample)
            proba = pipe.predict_proba(sample)
            self.assertIn(pred[0], ["Fatal", "Injury", "Property Damage"])
            self.assertEqual(len(proba[0]), 3)

    def test_map_assets_exist(self):
        hotspot = PROJECT_ROOT / "visualizations" / "maps" / "accident_hotspot_map.html"
        distrib = PROJECT_ROOT / "visualizations" / "maps" / "accident_distribution_map.html"
        self.assertTrue(hotspot.exists(), "Hotspot map HTML must exist")
        self.assertTrue(distrib.exists(), "Distribution map HTML must exist")

if __name__ == "__main__":
    unittest.main()
