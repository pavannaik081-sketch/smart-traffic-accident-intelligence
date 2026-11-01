import runpy
import sys
from pathlib import Path

# Add project root to sys.path so modules and paths resolve smoothly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main dashboard application
dashboard_app = PROJECT_ROOT / "dashboard" / "app.py"
runpy.run_path(str(dashboard_app), run_name="__main__")
